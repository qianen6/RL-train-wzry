import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual3DBlock(nn.Module):
    """带残差连接的3D卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))


class TemporalStateEncoder(nn.Module):
    """改进的时序编码器，带多头注意力"""

    def __init__(self, input_dim=200, hidden_dim=128, num_heads=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        output, _ = self.gru(x)  # [B,4,128]
        output = output.permute(1, 0, 2)  # [4,B,128]
        attn_output, _ = self.attn(output, output, output)
        return self.layer_norm(attn_output.mean(dim=0))  # [B,128]


class SpatiotemporalEncoder(nn.Module):
    """增强的时空编码器，带残差连接"""

    def __init__(self):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            Residual3DBlock(64, 128),
            nn.MaxPool3d((1, 2, 2)),
            Residual3DBlock(128, 256),
            nn.AdaptiveAvgPool3d((4, 36, 20))
        )

    def forward(self, x):
        return self.conv3d(x)


class CrossModalAttention(nn.Module):
    """跨模态注意力融合模块"""

    def __init__(self, visual_dim, state_dim):
        super().__init__()
        # 维度对齐设计
        self.state_dim = state_dim  # 保存 state_dim
        self.visual_proj = nn.Linear(visual_dim, state_dim)  # 视觉特征投影到状态维度
        self.state_proj = nn.Linear(256, state_dim)  # 降维
        self.query = nn.Linear(state_dim, state_dim)
        self.key = nn.Linear(state_dim, state_dim)

    def forward(self, visual_feat, state_feat):
        # visual_feat: [B, C, H, W]
        # state_feat: [B, D]
        # 步骤1：视觉特征维度投影
        B, C, H, W = visual_feat.shape  # C=256

        # 统一 state_feat 维度
        state_feat = self.state_proj(state_feat)  # 变成 state_dim 维（比如 128）

        visual_flat = visual_feat.permute(0, 2, 3, 1)  # [B, H, W, C]
        visual_projected = self.visual_proj(visual_flat)  # [B, H, W, state_dim]
        visual_projected = visual_projected.view(B, -1, self.state_dim)  # [B, H*W, state_dim]

        state_query = self.query(state_feat).unsqueeze(2)  # [B, state_dim, 1]

        attn_scores = torch.bmm(visual_projected, state_query)  # [B, H*W, 1]
        attn_weights = F.softmax(attn_scores.squeeze(-1), dim=1)  # [B, H*W]

        visual_processed = (visual_projected * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, state_dim]
        return visual_processed


class HybridNet(nn.Module):
    def __init__(self, left_dim=3, right_dim=3):
        super().__init__()
        self.gray_encoder = SpatiotemporalEncoder()
        self.minimap_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 2)),
            nn.Flatten(),
        )
        self.state_encoder = TemporalStateEncoder()
        self.minimap_proj = nn.Linear(1024, 128)  # 让minimap_feat匹配state_feat

        self.cross_attn = CrossModalAttention(256, 128)
        self.fusion = nn.Sequential(
            nn.Conv3d(256, 512, (4, 1, 1)),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((18, 10)),
            nn.Flatten(),
            nn.Dropout(0.05),
            nn.Linear(46080, 4096)
        )

        self.left_head = DuelingHead(4224, left_dim)
        self.right_head = DuelingHead(4224, right_dim)

    def forward(self, image1, image2, states):
        gray_feat = self.gray_encoder(image1.unsqueeze(1))
        image2 = image2.permute(0, 3, 1, 2)
        minimap_feat = self.minimap_encoder(image2)
        minimap_feat = self.minimap_proj(minimap_feat)  # 降维到 [B,128]

        state_feat = self.state_encoder(states)
        gray_flat = gray_feat.mean(dim=2)

        fused_visual = self.cross_attn(gray_flat, torch.cat([state_feat, minimap_feat], dim=1))

        final_feat = self.fusion[0](gray_feat).squeeze(2)
        final_feat = self.fusion[1:](final_feat)

        final_feat = torch.cat([final_feat, fused_visual + minimap_feat], dim=1)

        return self.left_head(final_feat), self.right_head(final_feat)


class DuelingHead(nn.Module):
    """增强的Dueling Head"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        return V + (A - A.mean(dim=1, keepdim=True))



