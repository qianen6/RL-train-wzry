import numpy as np
import cv2
from typing import Tuple
from collections import deque
from new_wzry_ai.config.default_config import GameConfig

class FrameStack:
    def __init__(self, 
                 stack_size: int = GameConfig.FRAME_STACK_SIZE,
                 frame_size: Tuple[int, int] = (GameConfig.FRAME_HEIGHT, GameConfig.FRAME_WIDTH)):
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.frames = deque(maxlen=stack_size)
        self.vectors = deque(maxlen=stack_size)
        
    def reset(self) -> None:
        """重置帧缓存"""
        self.frames.clear()
        self.vectors.clear()
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理单帧图像"""
        if not GameConfig.CorlorImage:
            # 转换为灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 调整大小,保持16:9比例
        # frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]),
        #                     interpolation=cv2.INTER_AREA)
        # 归一化到[0,1]范围
        normalized = frame.astype(np.float32) / 255.0
        
        return normalized
        
    def add_frame(self, frame: np.ndarray) -> np.ndarray:
        """添加新帧并返回叠加结果"""
        processed = self._preprocess_frame(frame)
        self.frames.append(processed)
        
        # 如果帧数不足，用第一帧填充
        while len(self.frames) < self.stack_size:
            self.frames.append(processed)

        images= np.stack(self.frames, axis=0)
        print(images.shape)
        # 将帧堆叠为3D数组
        return images

    def add_vector(self, vector: np.ndarray) -> np.ndarray:#堆叠状态向量
        """添加新帧并返回叠加结果"""
        self.vectors.append(vector)

        # 如果帧数不足，用第一帧填充
        while len(self.vectors) < self.stack_size:
            self.vectors.append(vector)

        vectors = np.stack(self.vectors, axis=0)
        print(vectors.shape)
        # 将帧堆叠为3D数组
        return vectors