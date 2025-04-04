"""
深度Q网络模型定义
"""
import torch
from torch import nn
from typing import Tuple

from ultralytics.nn.modules import ChannelAttention
from new_wzry_ai.utils.PrintUtils import print_utils



class DQN(nn.Module):
    def __init__(self, left_output_dim: int,right_output_dim: int):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            SpatialAttention()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            SpatialAttention()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            SpatialAttention()
        )

         # LSTM层[batch_size, seq_len, input_size]
        self.lstm = nn.LSTM(input_size=6272, hidden_size=512, batch_first=True)

        # 左手动作
        self.fc_left = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, left_output_dim)
        )  # (batch_size, 32, 34, 62) -> (batch_size, 64, 16, 30) -> (batch_size, 64, 14, 28)

        # 右手动作
        self.fc_right = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, right_output_dim)
        )  # (batch_size, 32, 34, 62) -> (batch_size, 64, 16, 30) -> (batch_size, 64, 14, 28)

    def forward(self, image1: torch.Tensor,image2:torch.Tensor,states_vectors:torch.Tensor):
        """
        image1:游戏全局截图
        image2：游戏小地图
        states_vectors:状态向量，包含各种局内信息状态
        """
        image1 = self.conv1(image1)

        image1 = self.conv2(image1)

        image1 = self.conv3(image1)
        image1 = image1.view(image1.size(0), 4, -1)  # [batch_size,seq_len=4, features]展平

        # 将展平的输出作为序列输入到LSTM
        # 假设每个时间步是一个特征向量

        # LSTM层LSTM 需要的数据格式是 [batch_size, seq_len, features]
        lstm_out, _ = self.lstm(image1)  # 输出：[batch_size, seq_len=4, hidden_size]

        # 只使用LSTM最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        left_output = self.fc_left(lstm_out)

        right_output = self.fc_right(lstm_out)
        return left_output , right_output


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_attention = torch.cat([avg_out, max_out], dim=1)

        attention = torch.sigmoid(self.conv(x_attention))

        return x * attention.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


class HybridNet(nn.Module):
    def __init__(self, left_output_dim: int,right_output_dim: int):
        super().__init__()
        # 单帧特征提取 (保留颜色)
        """
        frame_encoder 用于处理每一帧输入图像，提取每一帧的特征。
        nn.Conv2d(3, 32, kernel_size=5, stride=2)：第一层卷积，用于提取低级的空间特征，输出的通道数为 32，卷积核大小为 5x5，步幅为 2。
        CBAM(32)：应用 CBAM 模块，对卷积输出进行通道和空间上的加权。
        nn.Conv2d(32, 64, kernel_size=3)：进一步的卷积操作，将通道数从 32 增加到 64，卷积核大小为 3x3。
        nn.Flatten()：将输出的特征图展平成一维向量，准备输入到后续的 LSTM 层。
        """
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=8),#处理灰度图像
            nn.ReLU(),
            CBAM(32),
            nn.Conv2d(32, 64, kernel_size=4,stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            SpatialAttention(),
            nn.Flatten()
        )

        # 状态特征向量
        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # 拼接后的全连接层
        self.fc_combined = nn.Sequential(
            nn.Linear(1088, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        # 时序建模
        self.lstm = nn.LSTM(input_size=4352, hidden_size=512,batch_first=True)


        # 左手动作u
        self.fc_left = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, left_output_dim)
        )

        # 右手动作
        self.fc_right = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, right_output_dim)
        )  # (batch_size, 32, 34, 62) -> (batch_size, 64, 16, 30) -> (batch_size, 64, 14, 28)

    def forward(self,image1,image2:torch.Tensor,states_vectors:torch.Tensor):
        """
        iamge1--torch.Size([32, 4, 720, 1600])
        iamge2--torch.Size([32, 256, 256, 3])
        states_vectors--torch.Size([32, 4, 200])
        """
        batch_size,seq_len = image1.shape[:2]
        # 逐帧编码
        features = []
        for t in range(seq_len):
            frame = image1[:, t].unsqueeze(1)  # [32,1,720,1600]
            frame_feat = self.frame_encoder(frame)
            state_feat = self.fc(states_vectors[:,t])
            combined = torch.cat((frame_feat, state_feat), dim=1)
            features.append(combined)

        # 堆叠时间序列
        features = torch.stack(features, dim=1)  # [32, 4, 4032]
        #features = features.unsqueeze(1)  # 添加维度后形状变为 [32, 1, 401280]
        # LSTM聚合
        lstm_out, _ = self.lstm(features)  # [batch, seq_len, 512]
        image2feature = self.conv(image2.permute(0, 3, 1, 2))
        # 将LSTM输出和image2特征拼接
        lstm_out = lstm_out[:, -1, :]  # 取LSTM最后一个时间步的输出 [32, 512]
        combined_features = torch.cat((lstm_out, image2feature), dim=1)  # [32, 512 + 64*7*7]
        # 通过全连接层
        fc_output = self.fc_combined(combined_features)  # [32, 512]
        left_output = self.fc_left(fc_output)
        right_output = self.fc_right(fc_output)
        return left_output,right_output