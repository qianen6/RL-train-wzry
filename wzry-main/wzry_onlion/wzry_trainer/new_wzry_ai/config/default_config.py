from dataclasses import dataclass
from typing import Final
@dataclass(frozen=True)
class TrainingConfig:
    """训练相关配置"""
    BATCH_SIZE: Final[int] = 8#批次大小
    LEARNING_RATE: Final[float] = 0.0001#学习率
    GAMMA: Final[float] = 0.99#折扣因子
    EPSILON_START: Final[float] = 1.0#初始探索率
    EPSILON_MIN: Final[float] = 0.01#最小探索率
    EPSILON_DECAY: Final[float] = 0.9999#探索率衰减率
    UPDATE_TARGET_FREQ: Final[int] = 1000#更新目标网络频率
    MEMORY_SIZE: Final[int] = 100000#记忆库大小
    ALPHA: Final[float] = 0.6#优先级采样参数
    BETA_START: Final[float] = 0.4#重要性采样参数
    BETA_FRAMES: Final[int] = 100000#重要性采样帧数
    NUM_EPISODES: Final[int] = 10000#训练次数
    MAX_STEPS: Final[int] = 3000#最大步数
    SAVE_FREQ: Final[int] = 1#保存频率
    MODEL_PATH: Final[str] = '../models/wzry_model.pth'#模型保存路径
    YOLO_MODEL_PATH = 'runs/detect/train/weights/best.pt'#yolo模型路径
    left_action_states:int = 9,
    right_action_states: int = 8