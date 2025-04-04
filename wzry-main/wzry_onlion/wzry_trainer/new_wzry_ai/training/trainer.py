import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Event
from pynput import keyboard
from new_wzry_ai.core.agent import DoubleDQN
from new_wzry_ai.training.logger import TrainingLogger


from new_wzry_ai.config.default_config import (
    TrainingConfig,
)

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
            if hasattr(key, 'char') and key.char.lower() == 'q':
                self.logger.info("The Q key is detected and the training is ready to be stopped...")
                self.stop_event.set()
                self.episode_stop_event.set()
        except AttributeError:
            pass



    def _init_components(self):
        """初始化组件"""
        self.agent = DoubleDQN(action_dims=(TrainingConfig.left_action_states,TrainingConfig.right_action_states))

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


    def train(self):
        """训练主循环"""
        self.logger.info("Start training...")
        self.logger.info("Tip: Press the Q key to stop the training at any time")
        # 启动训练工作线程
        training_future = self.training_pool.submit(self._training_worker)
        try:
            while not self.stop_event.is_set():
                pass
                # self.agent.save_model(self.config.model_path)
        except Exception as e:
            self.logger.error("There was an error during the training process", exc_info=e)
        finally:
            if training_future.running():
                training_future.cancel()
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

