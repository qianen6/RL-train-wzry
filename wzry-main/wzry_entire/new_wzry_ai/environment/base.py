from typing import Tuple, Dict, Any, Optional
import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass
from new_wzry_ai.config.default_config import GameConfig

@dataclass
class EnvState:
    current_frame: Optional[np.ndarray] = None
    episode_step: int = 0
    current_min_map: Optional[np.ndarray] =None

class BaseEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        """初始化环境"""
        super().__init__()
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(GameConfig.FRAME_STACK_SIZE, 
                   GameConfig.FRAME_HEIGHT, 
                   GameConfig.FRAME_WIDTH),
            dtype=np.float32
        )
        
        # 定义动作空间
        self.action_space = None  # 由子类实现
        
        # 环境状态
        self.state = EnvState()
        
    def step(self, action,episodes) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError
        
    def reset(self) -> np.ndarray:
        raise NotImplementedError
        
    def render(self, mode='human'):
        """渲染当前环境状态"""
        pass
        
    def close(self):
        """关闭环境"""
        pass 