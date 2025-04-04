import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from new_wzry_ai.config.default_config import TrainingConfig
from threading import Lock

@dataclass
class Experience:
    state: list
    action: Tuple[int, int]
    reward: float
    next_state: list
    done: bool

class TDReplayBuffer:
    def __init__(self, 
                 capacity: int = TrainingConfig.MEMORY_SIZE,
                 alpha: float = TrainingConfig.ALPHA,
                 beta_start: float = TrainingConfig.BETA_START,
                 beta_frames: int = TrainingConfig.BETA_FRAMES,
                 gamma: float = TrainingConfig.GAMMA):
        self.capacity = capacity  # 缓存容量
        self.alpha = alpha  # 优先级采样超参数
        self.beta_start = beta_start  # beta 初始值
        self.beta_frames = beta_frames  # beta 增长帧数
        self.gamma = gamma  # 折扣因子（TD 误差计算使用）
        self.frame = 1
        self.buffer: List[Experience] = []  # 经验缓存
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 优先级
        self.buffer_lock = Lock()
        
    def beta_by_frame(self, frame_idx: int) -> float:
        """计算当前帧的 beta 值"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action: Tuple[int, int], reward: float,
             next_state, done: bool, td_error: float = None) -> None:
        with self.buffer_lock:
            experience = Experience(state, action, reward, next_state, done)

            # 如果没有 TD 误差，则使用最大优先级
            max_prio = self.priorities.max() if self.buffer else 1.0
            prio = abs(td_error) + 1e-5 if td_error is not None else max_prio

            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.pos] = experience
                
            self.priorities[self.pos] = prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        with self.buffer_lock:
            if len(self.buffer) == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:self.pos]

            if prios.sum() == 0 or np.isnan(prios.sum()):  # 避免出现 NaN 或全 0
                prios = np.ones_like(prios)  # 设为均匀分布，避免 NaN

            # 计算采样概率
            probs = prios ** self.alpha
            probs /= probs.sum()  # 归一化概率

            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]

            # 计算重要性权重
            beta = self.beta_by_frame(self.frame)
            self.frame += 1

            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)

            # 解包样本
            images1 = np.array([exp.state[0] for exp in samples], dtype=np.float32)
            images2 = np.array([exp.state[1] for exp in samples], dtype=np.float32)
            states_vectors = np.array([exp.state[2] for exp in samples], dtype=np.float32)
            actions = np.array([exp.action for exp in samples])
            rewards = np.array([exp.reward for exp in samples], dtype=np.float32)
            next_images1 = np.array([exp.next_state[0] for exp in samples], dtype=np.float32)
            next_images2 = np.array([exp.next_state[1] for exp in samples], dtype=np.float32)
            next_states_vectors = np.array([exp.next_state[2] for exp in samples], dtype=np.float32)
            dones = np.array([exp.done for exp in samples], dtype=np.float32)

            return images1, images2, states_vectors, actions, rewards, next_images1, next_images2, next_states_vectors, dones, indices, weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """根据 TD 误差更新优先级"""
        with self.buffer_lock:
            for idx, td_error in zip(indices, td_errors):
                if np.isnan(td_error) or np.isinf(td_error):  # 避免无效数值
                    td_error = 1.0  # 设置为一个合理的默认值

                self.priorities[idx] = abs(td_error) + 1e-5  # 确保不会出现 0

    def memory_buffer_len(self) -> int:
        """返回当前缓存中的样本数量"""
        return len(self.buffer)

# 经验回放缓存,实例化
memory = TDReplayBuffer()
