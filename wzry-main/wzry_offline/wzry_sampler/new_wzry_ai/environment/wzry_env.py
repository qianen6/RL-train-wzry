from typing import Tuple, Dict, Any, Optional
import numpy as np
import time
import cv2
from gym import spaces
from new_wzry_ai.environment.base import BaseEnvironment, EnvState
from new_wzry_ai.environment.reward import RewardCalculator
from new_wzry_ai.environment.exceptions import GameNotReadyError
from new_wzry_ai.utils.screen import ScreenCapture
from new_wzry_ai.utils.frame_stack import FrameStack
from new_wzry_ai.utils.game_controller import ActionExecutor
from new_wzry_ai.config.default_config import GameConfig
from new_wzry_ai.config.default_config import other_status, TemplateRegion
from new_wzry_ai.utils.heros_position_detector import HeroPositionDetector
from new_wzry_ai.utils.state_vector import StateVector


class WzryEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__()
        
        # 初始化组件
        self.screen = ScreenCapture()
        self.reward_calculator = RewardCalculator()
        self.action_executor = ActionExecutor()
        self.frame_stacker = FrameStack()
        # 状态向量
        self.state_vector = StateVector()
        # 定义动作空间
        self.left_action_space = spaces.Discrete(9)   # 8个移动动作 + 1个不动作 left=9
        self.right_action_space = spaces.Discrete(8)  # 7个技能动作 + 1个不动作 right=8
        self.action_space = spaces.Tuple((
            self.left_action_space,
            self.right_action_space
        ))
        self.hero_position_detector = HeroPositionDetector()

    def process_minimap(self, min_map):
        # 计算缩放目标尺寸
        target_width = min_map.shape[1] // 4  # 宽度缩小4倍   64 64
        target_height = min_map.shape[0] // 4  # 高度缩小4倍

        # 拆分颜色通道
        b, g, r = cv2.split(min_map)

        # 合并红色和蓝色通道
        red_blue_img = np.stack([b, r], axis=2)  # (H, W, 2)

        # 缩放至目标尺寸
        red_blue_img = cv2.resize(red_blue_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # 转换类型并归一化
        red_blue_img_float = red_blue_img.astype(np.float32)
        red_blue_img_normalized = red_blue_img_float / 255.0

        return red_blue_img_normalized

    def step(self, action: Tuple[int, int], episodes: int = 1):

        """执行一步动作"""
        left_action, right_action = action
        
        # 执行动作并等待完成
        action_start_time = time.time()


        self.action_executor.execute_action_async(left_action, right_action)
        time.sleep(GameConfig.FRAME_TIME)
        action_end_time = time.time()
        print(f"执行时间: {action_end_time - action_start_time:.6f} 秒")

        observation_start_time = time.time()
        # 获取新的观察
        self.state.current_frame = self.screen.capture()  # 捕获游戏画面帧

        x1, y1, x2, y2 = TemplateRegion.CHARACTER_AREA

        self.state.min_map = self.state.current_frame[y1:y2, x1:x2]  # 获取小地图

        min_map = self.process_minimap(self.state.min_map)

        if self.state.current_frame is None:
            raise GameNotReadyError("无法获取游戏画面")
        
        # 处理观察并计算奖励
        observation = self.frame_stacker.add_frame(self.state.current_frame)  # 堆叠四帧图像

        observation_end_time = time.time()
        print(f"观察时间: {observation_end_time - observation_start_time:.6f} 秒")

        infer_start_time = time.time()  # 记录开始时间
        reward, done = self.reward_calculator.calculate_total_reward(
            self.state.current_frame, right_action, left_action, episodes, self.state.min_map
        )
        infer_end_time = time.time()  # 记录结束时间
        print(f"reward时间: {infer_end_time - infer_start_time:.6f} 秒")

        #更新状态向量
        self.state_vector.update(self.state.current_frame, action, reward)

        # 更新步数
        self.state.episode_step += 1
        
        # 获取当前状态信息
        info = self._get_state_info()

        vector = self.state_vector.get_state()

        vectors = self.frame_stacker.add_vector(vector)

        next_state = [observation, min_map, vectors]



        return next_state, reward, done, info

    def reset_character_statue(self):
        other_status['if_alive'] = True
        other_status['if_gae_start'] = False

    def reset(self):
        """重置环境"""
        self.state = EnvState()
        self.frame_stacker.reset()
        
        # 重置所有按键状态
        self.action_executor.execute_action_async(8, 7)

        # 重置角色状态
        self.reset_character_statue()

        # 获取初始观察
        self.state.current_frame = self.screen.capture()

        x1, y1, x2, y2 = TemplateRegion.CHARACTER_AREA

        self.state.min_map = self.state.current_frame[y1:y2, x1:x2]  # 获取小地图256 256

        min_map = self.process_minimap(self.state.min_map)
        if self.state.current_frame is None:
            raise GameNotReadyError("无法获取游戏画面")

        observation=self.frame_stacker.add_frame(self.state.current_frame)

        #更新状态向量
        self.state_vector.update(self.state.current_frame, (8,7), 0)

        state_vector = self.state_vector.get_state()
        vectors = self.frame_stacker.add_vector(state_vector)

        next_state = [observation, min_map, vectors]
        return next_state

    def close(self):
        """关闭环境"""
        # 释放所有按键
        self.action_executor.execute_action_async(8, 7)
        self.action_executor.close()

    def _get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        current_blood = self.reward_calculator.blood_detector.get_self_blood(
            self.state.current_frame
        )
        enemy_blood_changed = self.reward_calculator.blood_detector.is_enemy_blood_changed(
            self.state.current_frame
        )
        
        current_pos = self.reward_calculator.routing_guide.get_houyi_position(
            self.state.current_frame
        )
        is_at_dest, dest_name = self.reward_calculator.routing_guide.is_at_any_destination(
            self.state.current_frame, current_pos
        )
        
        attacking_monster = self.reward_calculator.routing_guide.detect_any_monster(
            self.state.current_frame
        )
            
        return {
            "episode_step": self.state.episode_step,
            "self_blood": current_blood,
            "enemy_blood_changed": enemy_blood_changed,
            "current_pos": current_pos,
            "is_at_dest": is_at_dest,
            "dest_name": dest_name if is_at_dest else "无",
            "attacking_monster": attacking_monster
        }
