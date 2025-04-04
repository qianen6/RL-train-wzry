import os
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Event

from matplotlib import pyplot as plt
from pynput import keyboard
from new_wzry_ai.core.agent import DoubleDQN
from new_wzry_ai.training.logger import TrainingLogger
from new_wzry_ai.core.memory import memory
import torch
from torch.utils.tensorboard import SummaryWriter

from new_wzry_ai.config.default_config import TrainingConfig


@dataclass
class TrainerConfig:
    load_model: bool = False
    model_path: str = TrainingConfig.MODEL_PATH
    save_freq: int = TrainingConfig.SAVE_FREQ


class WzryTrainer:
    def __init__(self, config: TrainerConfig = TrainerConfig()):
        self.config = config
        self.logger = TrainingLogger()
        # 创建线程池和队列
        self.training_pool = ThreadPoolExecutor(max_workers=3)
        self.training_queue = Queue(maxsize=1)

        # 停止事件
        self.stop_event = Event()
        self.episode_stop_event = Event()

        # 初始化键盘监听器
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()

        self.logger.info("Initialize the trainer...")
        self._init_components()

    def _on_key_press(self, key):
        """键盘按键回调函数"""
        try:
            # 检查是否按下Q键
            if hasattr(key, 'char') and key.char == 'V':
                self.logger.info("The V key is detected and the training is ready to be stopped...")
                self.stop_event.set()
                self.episode_stop_event.set()
        except AttributeError:
            pass

    def _init_components(self):
        """初始化组件"""
        self.agent = DoubleDQN(action_dims=(TrainingConfig.left_action_states.n, TrainingConfig.right_action_states.n))

        # 创建模型保存目录
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        # 加载模型
        if self.config.load_model:
            if os.path.exists(self.config.model_path):
                self.logger.info(f"Load an existing model: {self.config.model_path}")
                self.agent.load_model(self.config.model_path)
            else:
                self.logger.warning(f"The model file does not exist: {self.config.model_path}")

        self.logger.info("Trainer initialization is complete!")


    def load_experiences(self,directory):
        if not os.path.exists(directory):
            print(f"经验文件目录 {directory} 不存在")
            return

        pth_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        if not pth_files:
            print("未找到任何 .pth 经验文件")
            return

        for pth_file in pth_files:
            file_path = os.path.join(directory, pth_file)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    loaded_experiences = torch.load(file_path)
                    for exp in loaded_experiences:
                        state, action, reward, next_state, done = exp  # 解包经验
                        memory.push(state, action, reward, next_state, done)
                    print(f"成功加载 {file_path}, 当前经验池大小：{memory.memory_buffer_len()}")
            except Exception as e:
                print(f"加载 {file_path} 时出错: {e}")

    def train(self):
        """训练主循环"""
        self.logger.info("Start training...")
        self.logger.info("Tip: Press the V key to stop the training at any time")

        losses_left = []
        losses_right = []#存储左右手loss
        # 启动训练工作线程
        #self.load_experiences(TrainingConfig.EXPERIENCES_FILEPATH)
        self.load_experiences("../experiences/experience3/")
        #self.load_experiences("../experiences/experience2/")
        #self.load_experiences("../experiences/experience3/")
        count1=0
        count2=0
        try:
            while not self.stop_event.is_set():
                loss=self.agent.learn()
                if loss is not None:
                    loss_left, loss_right = loss
                    losses_left.append(loss_left)
                    losses_right.append(loss_right)
                count2+=1
                if count2==100:
                    count1+=1
                    count2=0

                    # 绘制 loss 曲线
                    plt.figure(figsize=(8, 5))
                    plt.plot(losses_left, label="Loss Left Hand")
                    plt.plot(losses_right, label="Loss Right Hand")
                    plt.xlabel("Training Steps")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.title("Training Loss Curve")

                    plt.savefig(TrainingConfig.LOSS_IMAGE_DIR)  # 保存图像
                    plt.close()
                    print(f"批次大小为{TrainingConfig.BATCH_SIZE},当前已训练{count1}H批次")

        except Exception as e:
            self.logger.error("There was an error during the training process", exc_info=e)
        finally:
            self.logger.info("The final model and statistics are being saved...")
            self.agent.save_model(self.config.model_path)

    def close(self):
        """关闭训练器"""
        self.stop_event.set()
        self.episode_stop_event.set()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        self.training_pool.shutdown(wait=True)

    def __del__(self):
        """析构函数"""
        self.close()
