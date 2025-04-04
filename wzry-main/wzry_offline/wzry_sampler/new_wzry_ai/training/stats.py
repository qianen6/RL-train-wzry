import os
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class TrainingStats:
    total_episodes: int = 0
    victories: int = 0
    defeats: int = 0
    win_rate: float = 0.0
    total_rewards: float = 0.0
    avg_reward: float = 0.0
    total_steps: int = 0
    avg_steps: float = 0.0
    
    def update(self, reward: float, steps: int, victory: bool = None):
        self.total_episodes += 1
        self.total_rewards += reward
        self.total_steps += steps
        
        if victory is not None:
            if victory:
                self.victories += 1
            else:
                self.defeats += 1
                
        self.win_rate = self.victories / max(1, self.total_episodes)
        self.avg_reward = self.total_rewards / self.total_episodes
        self.avg_steps = self.total_steps / self.total_episodes

class StatsRecorder:
    def __init__(self, save_dir: str = "stats"):
        self.stats = TrainingStats()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def update(self, reward: float, steps: int, victory: bool = None):
        """更新统计数据"""
        self.stats.update(reward, steps, victory)
        
    def save(self, filename: str = "training_stats.json"):
        save_path = os.path.join(self.save_dir, filename)
        with open(save_path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=4)
            
    def load(self, filename: str = "training_stats.json") -> bool:
        load_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(load_path):
            return False
            
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
                self.stats = TrainingStats(**data)
            return True
        except Exception:
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据字典"""
        return asdict(self.stats) 