"""
优先经验回放缓存实现
"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from new_wzry_ai.config.default_config import TrainingConfig
from threading import Lock

@dataclass
class Experience:
    state:list
    action: Tuple[int, int]
    reward: float
    next_state:list
    done: bool

class PriorityReplayBuffer:
    def __init__(self, 
                 capacity: int = TrainingConfig.MEMORY_SIZE,
                 alpha: float = TrainingConfig.ALPHA,
                 beta_start: float = TrainingConfig.BETA_START,
                 beta_frames: int = TrainingConfig.BETA_FRAMES):
        self.capacity = capacity# 缓存容量
        self.alpha = alpha# 优先级采样的超参数
        self.beta_start = beta_start# beta 的初始值
        self.beta_frames = beta_frames# beta 的增长帧数
        self.frame = 1
        self.buffer: List[Experience] = []# 经验缓存
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)# 优先级
        self.buffer_lock = Lock()
        
    def beta_by_frame(self, frame_idx: int) -> float:
        """计算当前帧的beta值"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        
    def push(self, state, action: Tuple[int, int], reward: float,
             next_state, done: bool) -> None:
        with self.buffer_lock:
            experience = Experience(state, action, reward, next_state, done)
            # 检查 self.buffer 是否为空。如果 self.buffer 非空，则将 max_prio 设置为 self.priorities 数组中的最大值；
            # 如果 self.buffer 为空，则将 max_prio 设置为 1.0。
            max_prio = self.priorities.max() if self.buffer else 1.0
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.pos] = experience
                
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
            
    def sample(self, batch_size: int) -> Tuple:
        with self.buffer_lock:
            if len(self.buffer) == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:self.pos]
                
            # 计算采样概率
            probs = prios ** self.alpha
            probs /= probs.sum()
            
            # 采样索引
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)# 从 self.buffer 中采样 batch_size 个样本，在这个例子中，probs 是一维的，表示每个经验被采样的概率。
            samples = [self.buffer[idx] for idx in indices]# 根据索引获取样本

            # 计算重要性权重
            beta = self.beta_by_frame(self.frame)
            self.frame += 1

            weights = (len(self.buffer) * probs[indices]) ** (-beta)# 计算重要性权重，其中 beta 是一个随着时间增长的值，用于控制重要性权重的调整。这里的计算是为了让重要性权重更加平滑。
            weights /= weights.max()# 为了避免梯度爆炸，将重要性权重归一化
            weights = np.array(weights, dtype=np.float32)
            # 解包样本
            images1 = np.array([exp.state[0] for exp in samples], dtype=np.float32)#
            images2 = np.array([exp.state[1] for exp in samples], dtype=np.float32)  #
            states_vectors = np.array([exp.state[2] for exp in samples], dtype=np.float32)  #
            actions = np.array([exp.action for exp in samples])
            rewards = np.array([exp.reward for exp in samples], dtype=np.float32)
            next_images1 = np.array([exp.next_state[0] for exp in samples], dtype=np.float32)  #
            next_images2 = np.array([exp.next_state[1] for exp in samples], dtype=np.float32)  #
            next_states_vectors = np.array([exp.next_state[2] for exp in samples], dtype=np.float32)  #
            dones = np.array([exp.done for exp in samples], dtype=np.float32)
            
            return images1,images2,states_vectors, actions, rewards, next_images1,next_images2,next_states_vectors, dones, indices, weights
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        with self.buffer_lock:
            for idx, prio in zip(indices, priorities):
                self.priorities[idx] = prio + 1e-5  
            
    def __len__(self) -> int:
        """返回当前缓存中的样本数量"""
        return len(self.buffer)

# 经验回放缓存,实例化
memory = PriorityReplayBuffer()