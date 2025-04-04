import os
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, wait
from queue import Queue, Full, Empty
from threading import Event
import threading
from pynput import keyboard
from new_wzry_ai.core.agent import DoubleDQN
from new_wzry_ai.environment.wzry_env import WzryEnvironment
from new_wzry_ai.environment.exceptions import GameNotReadyError
from new_wzry_ai.utils.template_match import TemplateMatcher
from new_wzry_ai.training.logger import TrainingLogger, LogConfig
from new_wzry_ai.training.stats import StatsRecorder
from new_wzry_ai.config.default_config import other_status
from new_wzry_ai.LLM_Utils.ChatGptTool import chatgpt_tool,update_movetarget
from new_wzry_ai.config.default_config import (
    TrainingConfig, 
    GameConfig,
    reward_config
)

@dataclass
class TrainerConfig:
    load_model: bool = False
    model_path: str = TrainingConfig.MODEL_PATH
    save_freq: int = TrainingConfig.SAVE_FREQ
    max_wait_time: int = GameConfig.MAX_WAIT_TIME
    respawn_wait_time: int = GameConfig.RESPAWN_WAIT_TIME

class WzryTrainer:
    def __init__(self, config: TrainerConfig = TrainerConfig()):
        self.config = config
        self.logger = TrainingLogger()
        self.stats_recorder = StatsRecorder()


        
        # 创建线程池和队列
        self.training_pool = ThreadPoolExecutor(max_workers=3)
        self.detection_pool = ThreadPoolExecutor(max_workers=1)
        self.chatgpt_pool = ThreadPoolExecutor(max_workers=1)

        self.training_queue = Queue(maxsize=1)

        # 停止事件
        self.stop_event = Event()
        self.episode_stop_event = Event() 
        
        # 初始化键盘监听器
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        
        self.logger.info("Initialize the trainer...")
        self._init_components()

        # 新增代码  添加一个标志来切换动作选择模式
        self.action_mode = "agent"  # "agent" 或 "player"
        self.key_pressed_move = None  # 用于保存当前按下的移动键
        self.key_pressed_attack = None  # 用于保存当前按下的攻击键
        # 你的动作映射规则
        # self.left_action = {'w': 0, 'a': 1, 's': 2, 'd': 3, 'wa': 4, 'wd': 5, 'sa': 6, 'sd': 7}
        # self.right_action = {'j': 0, 't': 1, 'e': 2, 'r': 3, 'b': 4, 'c': 5, 'f': 6}
        self.key_pressed_move = set()  # 用于记录当前按下的移动键
        self.key_pressed_attack = set()  # 用于记录当前按下的攻击键

        # 在初始化时启动监听输入的线程
        self.input_thread = threading.Thread(target=self.start_listening)
        self.input_thread.daemon = True  # 设置为守护线程，主程序退出时会自动退出
        self.input_thread.start()

    def get_key_name(self, key):
        if isinstance(key, keyboard.KeyCode):
            return key.char
        else:
            return str(key)



    def on_press(self, key):
        key_name = self.get_key_name(key)
        # 检查是否是移动按键
        if key_name in ['w', 'a', 's', 'd']:  # 按下移动键
            self.key_pressed_move.add(key_name)
        # 检查是否是攻击按键
        elif key_name in ['j', 't', 'e', 'r', 'b', 'c' , 'f']:  # 按下攻击键
            self.key_pressed_attack.add(key_name)


    def on_release(self, key):
        key_name = self.get_key_name(key)
        if key_name in self.key_pressed_move:
            self.key_pressed_move.remove(key_name)
        if key_name in self.key_pressed_attack:
            self.key_pressed_attack.remove(key_name)


        # 按下的键没有在后续的操作中继续被使用时退出监听
        if key == keyboard.Key.esc:
            return False  # 退出监听器

    def _get_player_action(self):
        # 组合移动键
        move_action = 8
        if 'w' in self.key_pressed_move and 'a' in self.key_pressed_move:
            move_action = 4
        elif 'w' in self.key_pressed_move and 'd' in self.key_pressed_move:
            move_action = 5
        elif 's' in self.key_pressed_move and 'a' in self.key_pressed_move:
            move_action = 6
        elif 's' in self.key_pressed_move and 'd' in self.key_pressed_move:
            move_action = 7
        elif 'w' in self.key_pressed_move:
            move_action = 0
        elif 'a' in self.key_pressed_move:
            move_action = 1
        elif 's' in self.key_pressed_move:
            move_action = 2
        elif 'd' in self.key_pressed_move:
            move_action = 3

        # 组合攻击键
        attack_action = 7
        if 'j' in self.key_pressed_attack:
            attack_action = 0
        elif 't' in self.key_pressed_attack:
            attack_action = 1
        elif 'e' in self.key_pressed_attack:
            attack_action = 2
        elif 'r' in self.key_pressed_attack:
            attack_action = 3
        elif 'b' in self.key_pressed_attack:
            attack_action = 4
        elif 'c' in self.key_pressed_attack:
            attack_action = 5
        elif 'f' in self.key_pressed_attack:
            attack_action = 6


        return move_action, attack_action

    def start_listening(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

        
    def _on_key_press(self, key):
        """键盘按键回调函数"""
        try:
            # 检查是否按下Q键
            if hasattr(key, 'char') and key.char.lower() == 'q':
                self.logger.info("The Q key is detected and the training is ready to be stopped...")
                self.stop_event.set()
                self.episode_stop_event.set()

            # 新增代码
            elif key.char.lower() == 'l':
                self.logger.info("Switching to agent action selection mode...")
                self.action_mode = "agent"  # 切换到智能体模式
            elif key.char.lower() == 'm':
                self.logger.info("Switching to player action selection mode...")
                self.action_mode = "player"  # 切换到玩家模式
            # 新增代码

        except AttributeError:
            pass



    def _init_components(self):
        """初始化组件"""
        self.env = WzryEnvironment()
        self.agent = DoubleDQN(
            state_dim=(GameConfig.FRAME_STACK_SIZE,
                      GameConfig.FRAME_HEIGHT,
                      GameConfig.FRAME_WIDTH),#帧栈大小，帧高度，帧宽度，四张帧堆叠输入
            action_dims=(self.env.left_action_space.n,
                        self.env.right_action_space.n)
        )

        # 创建模型保存目录
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        # 加载模型
        if self.config.load_model:
            if os.path.exists(self.config.model_path):
                self.logger.info(f"Load an existing model: {self.config.model_path}")
                self.agent.load_model(self.config.model_path)
            else:
                self.logger.warning(f"The model file does not exist: {self.config.model_path}")

        # 初始化模板匹配器
        self.template_matcher = TemplateMatcher()

        self.logger.info("Trainer initialization is complete!")

    def wait_for_game_start(self) -> bool:
        """等待游戏开始"""
        self.logger.info("Wait for the game to start...")
        start_time = time.time()

        while time.time() - start_time < self.config.max_wait_time:
            image = self.env.screen.capture()
            # =================== test =============================

            # =================== test =============================
            if image is None:
                self.logger.info("image is None")
                continue
            if self.template_matcher.match_template(image, 'game_start').success:
                self.logger.info("Game start detected")
                other_status['if_game_start']=True
                time.sleep(1)  # 等待游戏完全加载
                return True

            time.sleep(1)

        self.logger.warning("Wait for the game to start timeout")
        return False

    def wait_for_respawn(self) -> Tuple[bool, Optional[bool]]:
        """等待角色重生"""
        start_time = time.time()

        while time.time() - start_time < self.config.respawn_wait_time:
            image = self.env.screen.capture()
            if image is None:
                continue

            # 检查游戏是否结束
            if self.template_matcher.match_template(image, 'victory').success:
                self.logger.info("Game wins detected")
                return False, True
            if self.template_matcher.match_template(image, 'defeat').success:
                self.logger.info("A game failure was detected")
                return False, False

            # 检查重生
            blood_template_exists = self.template_matcher.match_template(
                image, 'self_blood'
            ).success
            current_blood = self.env.reward_calculator.blood_detector.get_self_blood(
                image
            )
            print(blood_template_exists)
            print(current_blood)
            if blood_template_exists and current_blood > 0:
                time.sleep(0.5)

                confirm_image = self.env.screen.capture()
                if confirm_image is not None:
                    blood_template_exists_2 = self.template_matcher.match_template(
                        confirm_image, 'self_blood'
                    ).success
                    current_blood_2 = self.env.reward_calculator.blood_detector.get_self_blood(
                        confirm_image
                    )

                    if blood_template_exists_2 and current_blood_2 > 0:
                        self.logger.info(f"Character respawned detected (HP: {current_blood_2:.2f})")
                        other_status['if_alive']=True
                        return True, None

            time.sleep(1)

        self.logger.warning("Waiting for respawn timed out")
        return False, None

    def check_game_status(self, image) -> Tuple[bool, Optional[bool]]:
        """检查游戏状态"""
        try:


            if not other_status['if_alive']:
                self.logger.info("Character death detected")
                respawned, game_result = self.wait_for_respawn()
                if not respawned:
                    return False, game_result

            # 检查游戏是否结束
            if self.template_matcher.match_template(image, 'victory').success:
                self.logger.info("Game wins detected")
                return False, True
            if self.template_matcher.match_template(image, 'defeat').success:
                self.logger.info("A game failure was detected")
                return False, False

            return True, None
        except Exception as e:
            print("====================== error =======================")
            print(e)
            print("====================== error =======================")

    def _training_worker(self):
        """训练工作线程"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取训练数据
                training_data = self.training_queue.get(timeout=1)
                if training_data is None:
                    print("The data is empty")
                    continue

                # 执行训练
                print("Training")
                self.agent.learn()
                print("End of training")

            except Empty:
                print("Empty")

            except Exception as e:
                print("====================== error =======================")
                print(e)
                print("====================== error =======================")
                continue




    def train_episode(self , episodes: int) -> Tuple[float, int, Optional[bool]]:
        """训练一个回合(使用多线程)"""
        try:
            state = self.env.reset()
        except GameNotReadyError as e:
            self.logger.error("The game environment is not ready", exc_info=e)
            return 0.0, 0, None

        # 重置回合停止事件
        self.episode_stop_event.clear()

        # 启动训练工作线程
        training_future = self.training_pool.submit(self._training_worker)
        chatgpt_tool.clear_message()
        chatgpt_movetarget_future=self.chatgpt_pool.submit(update_movetarget)
        total_reward = 0
        steps = 0

        # 打印回合表头
        self._print_episode_header()

        # 初始化步骤开始时间
        step_start_time = time.time()
        game_time=step_start_time
        #初始化game_result
        game_result=None
        while not self.episode_stop_event.is_set():
            # 使用检测线程池处理游戏状态检测
            detection_future = self.detection_pool.submit(
                self.check_game_status, self.env.state.current_frame
            )

            # 新增代码 初始化左右手动作
            left_action = 8
            right_action = 7

            # 根据当前的 action_mode 选择动作
            if self.action_mode == "agent":
                # 使用智能体选择动作
                left_action, right_action = self.agent.select_action(state)
            elif self.action_mode == "player":
                # 使用玩家选择的动作
                left_action, right_action = self._get_player_action()  # 获取玩家选择的动作
            # 新增代码


            action = (left_action, right_action)

            try:
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
            except GameNotReadyError as e:
                self.logger.error("The game environment is not ready", exc_info=e)
                break

            # 计算当前步骤用时（毫秒）
            step_time = int((time.time() - step_start_time) * 1000)
            # 重置步骤开始时间
            step_start_time = time.time()

            # 存储经验
            self.agent.memory.push(state, action, reward, next_state, done)

            # 将训练数据放入队列
            try:
                self.training_queue.put_nowait(1)
            except Full:
                print("FULL")

            # 更新状态
            state = next_state
            total_reward += reward
            steps += 1

            # 打印步骤信息（使用实际步骤用时）
            self._print_step_info(steps, left_action, right_action, info, total_reward, step_time)

            # 等待检测结果
            game_continue, game_result = detection_future.result()
            other_status["time"]=time.time()-game_time
            if not game_continue:
                other_status['if_game_start']=False
                break

            if done:
                other_status['if_game_start'] = False
                break

        # 停止训练工作线程
        if training_future.running():
            training_future.cancel()
        # 停止chatgpt线程
        if chatgpt_movetarget_future.running():
            chatgpt_movetarget_future.cancel()

        return total_reward, steps, game_result
        # except Exception as e:
        #     print("====================== error =======================")
        #     print(e)
        #     print("====================== error =======================")

    def train(self):
        """训练主循环"""
        self.logger.info("Start training...")
        self.logger.info("Tip: Press the Q key to stop the training at any time")

        try:
            while not self.stop_event.is_set():
                self.logger.info("Wait for a new round of games to begin...")
                if not self.wait_for_game_start():
                    break
                if self.stop_event.is_set():
                    continue
                if not other_status['if_first_episode_have_start']:
                    other_status['if_first_episode_have_start']=True


                episode_start_time = time.time()
                self.logger.info(f"Round {self.stats_recorder.stats.total_episodes + 1} training begins")
                episodes=self.stats_recorder.stats.total_episodes + 1

                total_reward, steps, game_result = self.train_episode(episodes)
                episode_time = time.time() - episode_start_time

                if self.stop_event.is_set():
                    break

                # 更新统计信息
                self.stats_recorder.update(total_reward, steps, game_result)

                # 打印训练信息
                self._print_episode_summary(episode_time, steps, total_reward)

                # 保存模型和统计信息
                if self.stats_recorder.stats.total_episodes % self.config.save_freq == 0:
                    self.logger.info("Model and statistics are being saved...")
                    self.agent.save_model(self.config.model_path)
                    self.stats_recorder.save()

        except Exception as e:
            self.logger.error("There was an error during the training process", exc_info=e)
        finally:
            self.logger.info("The final model and statistics are being saved...")
            self.agent.save_model(self.config.model_path)
            self.stats_recorder.save()
            self.env.close()

            self._print_final_stats()

    def _print_episode_header(self):
        """打印回合表头"""
        print("\n" + "=" * 80)

    def _print_step_info(self, steps: int, left_action: int, right_action: int,
                        info: Dict[str, Any], total_reward: float, step_time: int):
        """打印步骤信息"""
        print(
            f"Time:{int(other_status['time'])}    |"
            f"Steps:{steps}    |   left_action：{left_action},rifht_action：{right_action}    | "
            f"my_blood：{info['self_blood']:.2f}  | current_target_position：{other_status['move_target']}   | "
            f"enemy：{'enemy blood changed' if info['enemy_blood_changed'] else 'not detected'} \n"
            f"if have monster：{'yes' if info['attacking_monster'] else 'no'}   | "
            #f"my position：{'not detect' if info['dest_name'] is None else info['dest_name']}     | "
            f"my position：{other_status['my_position']} | "
            f"reward：{total_reward:.2f}",
        )
        print(f"Situation analysis:{other_status['move_reason']}")


    def _print_episode_summary(self, episode_time: float, steps: int, total_reward: float):
        """打印回合总结"""
        stats = self.stats_recorder.stats
        print("\n\n" + "=" * 100)
        print(f"Summary of the  {stats.total_episodes} round:")
        print(f"  Training duration: {episode_time:.1f} seconds")
        print(f"  Total number of steps: {steps}")
        print(f"  Average time per step: {episode_time/steps:.3f} seconds")
        print(f"  Total Rewards:{total_reward:.2f}")
        print(f"  Exploration Rate: {self.agent.state.epsilon:.4f}")
        print(f"  Current Win Rate: {stats.win_rate:.2%}")
        print("=" * 100)

    def _print_final_stats(self):
        """打印最终统计信息"""
        stats = self.stats_recorder.stats
        print("\nEnd-of-training statistics:")
        print(f"  Total number of sessions: {stats.total_episodes}")
        print(f"  victory: {stats.victories}")
        print(f"  fail: {stats.defeats}")
        print(f"  Final win rate: {stats.win_rate:.2%}")

    def close(self):
        """关闭训练器"""
        self.stop_event.set()
        self.episode_stop_event.set()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        self.training_pool.shutdown(wait=True)
        self.detection_pool.shutdown(wait=True)

    def __del__(self):
        """析构函数"""
        self.close()

