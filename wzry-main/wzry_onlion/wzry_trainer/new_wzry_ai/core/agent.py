
import torch
from typing import Tuple, Optional
from dataclasses import dataclass
from threading import Lock
from new_wzry_ai.core.model import HybridNet
from new_wzry_ai.core.memory import memory
from new_wzry_ai.config.default_config import TrainingConfig
from new_wzry_ai.utils.PrintUtils import print_utils

@dataclass
class AgentState:
    epsilon: float
    steps_done: int

class DoubleDQN:
    def __init__(self,action_dims: Tuple[int, int] = (9, 8)):
        self.left_action_dim, self.right_action_dim = action_dims
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device:{self.device}")
        # 创建网络
        self.current_network = HybridNet(self.left_action_dim,self.right_action_dim).to(self.device)# 创建左手网络输出维度为9
        self.target_network = HybridNet(self.left_action_dim,self.right_action_dim).to(self.device)#创建目标左手网络输出维度为9
        
        # 初始化目标网络
        self.target_network.load_state_dict(self.current_network.state_dict())# 返回的是一个包含模型所有参数和缓冲区（例如 BatchNorm 中的 running_mean 和 running_var 等）状态的字典。
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.current_network.parameters(),#返回一个迭代器，包含所有可训练参数的 Tensor
            lr=TrainingConfig.LEARNING_RATE
        )

        # 损失函数
        self.loss_func = torch.nn.SmoothL1Loss(reduction='none')
        
        # 智能体状态
        self.state = AgentState(
            epsilon=TrainingConfig.EPSILON_START,
            steps_done=0
        )
        
        # 添加线程锁
        self.train_lock = Lock()
        self.action_lock = Lock()


    def learn(self) -> Optional[Tuple[float, float]]:

        with self.train_lock:
            if len(memory) < TrainingConfig.BATCH_SIZE:
                print_utils.print_green("None")
                return None

            # 采样数据

            image1s, image2s, states_vectors,actions, rewards, next_image1s,next_image2s,next_states_vectors, dones, indices, weights = self.memory.sample(TrainingConfig.BATCH_SIZE)


            #states是包含32个state的列表，每个state是一个包含3个元素的列表
            # 转换为tensor

            image1s = torch.tensor(image1s, dtype=torch.float32).to(self.device)
            image2s = torch.tensor(image2s, dtype=torch.float32).to(self.device)
            states_vectors = torch.tensor(states_vectors, dtype=torch.float32).to(self.device)
            left_actions = torch.tensor([a[0] for a in actions], dtype=torch.long).unsqueeze(-1).to(self.device)
            right_actions = torch.tensor([a[1] for a in actions], dtype=torch.long).unsqueeze(-1).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
            next_image1s = torch.tensor(next_image1s, dtype=torch.float32).to(self.device)
            next_image2s = torch.tensor(next_image2s, dtype=torch.float32).to(self.device)
            next_states_vectors = torch.tensor(next_states_vectors, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1).to(self.device)
            # 计算网络的损失
            current_q_left, current_q_right = self.current_network(image1s, image2s, states_vectors)
            current_q_left = current_q_left.gather(1, left_actions)
            current_q_right = current_q_right.gather(1, right_actions)

            with torch.no_grad():
                next_q_left, next_q_right = self.target_network(next_image1s,next_image2s,next_states_vectors)
                next_q_left = next_q_left.max(1, keepdim=True)[0]
                next_q_right = next_q_right.max(1, keepdim=True)[0]
                target_q_left = rewards + (1 - dones) * TrainingConfig.GAMMA * next_q_left
                target_q_right = rewards + (1 - dones) * TrainingConfig.GAMMA * next_q_right
            loss_left = (weights * self.loss_func(current_q_left, target_q_left)).mean()
            loss_right = (weights * self.loss_func(current_q_right, target_q_right)).mean()
            total_loss = loss_left + loss_right
            # 更新左右手网络
            self.optimizer.zero_grad()
            total_loss.backward()  # 计算所有层的梯度
            self.optimizer.step()

            # 更新优先级，移到GPU上进行计算，避免使用numpy
            new_priorities = torch.abs(
                (target_q_left - current_q_left) +
                (target_q_right - current_q_right)
            ).squeeze().detach().cpu().numpy()  # 使用 .detach() 断开梯度计算并转到 CPU
            memory.update_priorities(indices, new_priorities)

            # 更新探索率
            self.state.epsilon = max(
                TrainingConfig.EPSILON_MIN,
                self.state.epsilon - TrainingConfig.EPSILON_DECAY
            )

            # 更新目标网络
            if self.state.steps_done % TrainingConfig.UPDATE_TARGET_FREQ == 0:
                self.target_network.load_state_dict(self.current_network.state_dict())
                self.target_network.to(self.device)  # 确保目标网络在GPU上

            self.state.steps_done += 1
            print("success")
            return loss_left.item(), loss_right.item()

        # except Exception as e:
        #     print("====================== error =======================")
        #     print(e)
        #     print("====================== error =======================")

    def save_model(self, path: str) -> None:
        try:
            torch.save({
                'current_network_state_dict': self.current_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.state.epsilon,
                'steps_done': self.state.steps_done
            }, path, _use_new_zipfile_serialization=True)
            print(f"The model and training status are saved to {path}")
        except Exception as e:
            print(f"An exception occurred while saving the model: {e}")

    def load_model(self, path: str) -> None:
        try:
            checkpoint = torch.load(
                path, 
                map_location=self.device,
                weights_only=True
            )

            # 加载网络状态
            self.current_network.load_state_dict(checkpoint['current_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练状态
            self.state.epsilon = float(checkpoint['epsilon'])
            self.state.steps_done = int(checkpoint['steps_done'])
            
            print(f"The model and training status are loaded：{path}")
        except Exception as e:
            print(f"An exception occurred while loading the model: {e}")
            self._reset_model()
            
    def _reset_model(self) -> None:
        """重置模型到初始状态"""
        self.current_network = HybridNet(self.left_action_dim,self.right_action_dim).to(self.device)
        self.target_network = HybridNet(self.left_action_dim,self.right_action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.current_network.parameters(),
            lr=TrainingConfig.LEARNING_RATE
        )
        self.state.epsilon = TrainingConfig.EPSILON_START
        self.state.steps_done = 0
        print("Failed to load, model reinitialized")