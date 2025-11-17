"""
PatchTST Model Implementation
基于论文: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
参考实现: thuml/Time-Series-Library

核心创新：
1. Patching: 将长序列切分成patches，类似ViT处理图像
2. Channel Independence: 每个变量独立建模
3. Transformer: 使用自注意力机制捕获长期依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEmbedding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    """
    将时间序列切分成patches并编码

    Args:
        patch_len: 每个patch的长度
        stride: patch之间的步长
        d_model: Transformer的隐藏维度
        dropout: Dropout率
    """
    def __init__(self, patch_len=16, stride=8, d_model=128, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Patch embedding: 将每个patch映射到d_model维度
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len, n_vars]
        输出: [batch_size, n_vars, num_patches, d_model]
        """
        batch_size, seq_len, n_vars = x.shape

        # 计算patch数量
        num_patches = (seq_len - self.patch_len) // self.stride + 1

        # 为每个变量创建patches
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # [batch_size, patch_len, n_vars]
            patches.append(patch)

        # [batch_size, num_patches, patch_len, n_vars]
        patches = torch.stack(patches, dim=1)

        # 变换维度: [batch_size, n_vars, num_patches, patch_len]
        patches = patches.permute(0, 3, 1, 2)

        # Embedding: [batch_size, n_vars, num_patches, d_model]
        x_embed = self.value_embedding(patches)

        # 添加位置编码
        # 需要reshape来应用位置编码
        b, n, p, d = x_embed.shape
        x_embed_reshaped = x_embed.reshape(b * n, p, d)
        pos_embed = self.position_embedding(x_embed_reshaped)
        x_embed_reshaped = x_embed_reshaped + pos_embed
        x_embed = x_embed_reshaped.reshape(b, n, p, d)

        return self.dropout(x_embed), num_patches


class FlattenHead(nn.Module):
    """预测头"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [batch_size, n_vars, num_patches, d_model]
        输出: [batch_size, n_vars, target_window]
        """
        x = self.flatten(x)  # [batch_size, n_vars, num_patches * d_model]
        x = self.linear(x)    # [batch_size, n_vars, target_window]
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    PatchTST模型主体

    Args:
        input_dim: 输入特征数
        seq_len: 输入序列长度
        pred_len: 预测长度
        d_model: Transformer隐藏维度
        n_heads: 注意力头数
        e_layers: Encoder层数
        d_ff: FFN隐藏维度
        patch_len: Patch长度
        stride: Patch步长
        dropout: Dropout率
        target_dim: 目标维度（默认为1，只预测signal_1）
    """
    def __init__(
        self,
        input_dim=3,
        seq_len=2000,
        pred_len=1000,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=256,
        patch_len=16,
        stride=8,
        dropout=0.2,
        target_dim=1
    ):
        super(PatchTST, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.target_dim = target_dim

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout
        )

        # 计算patch数量
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # [seq, batch, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers
        )

        # Prediction Head (每个变量独立)
        self.head = FlattenHead(
            n_vars=input_dim,
            nf=self.num_patches * d_model,
            target_window=pred_len,
            head_dropout=dropout
        )

        # 如果只预测一个变量，添加一个选择层
        if target_dim == 1:
            self.output_projection = nn.Linear(input_dim, target_dim)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.0):
        """
        前向传播

        Args:
            x: [batch_size, seq_len, input_dim]
            targets: 用于兼容训练接口，PatchTST不使用teacher forcing
            teacher_forcing_ratio: 用于兼容，PatchTST不使用

        Returns:
            output: [batch_size, pred_len] (如果target_dim=1) 或 [batch_size, pred_len, target_dim]
        """
        batch_size = x.size(0)

        # Patch Embedding
        # x_embed: [batch_size, n_vars, num_patches, d_model]
        x_embed, num_patches = self.patch_embedding(x)

        # 处理每个变量（Channel Independence）
        outputs = []
        for i in range(self.input_dim):
            # 提取第i个变量: [batch_size, num_patches, d_model]
            x_i = x_embed[:, i, :, :]

            # Transformer期望输入: [num_patches, batch_size, d_model]
            x_i = x_i.permute(1, 0, 2)

            # Transformer Encoding
            enc_out = self.transformer_encoder(x_i)  # [num_patches, batch_size, d_model]

            # 转回: [batch_size, num_patches, d_model]
            enc_out = enc_out.permute(1, 0, 2)

            outputs.append(enc_out)

        # 堆叠: [batch_size, n_vars, num_patches, d_model]
        enc_out = torch.stack(outputs, dim=1)

        # Prediction Head
        # output: [batch_size, n_vars, pred_len]
        output = self.head(enc_out)

        # 如果只预测一个变量（signal_1）
        if self.target_dim == 1:
            # 取signal_1 (索引1)
            output = output[:, 1, :]  # [batch_size, pred_len]
        else:
            # [batch_size, pred_len, n_vars]
            output = output.permute(0, 2, 1)

        return output

    def get_num_params(self):
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())


def get_patchtst_model(
    input_dim=3,
    seq_len=2000,
    pred_len=1000,
    d_model=128,
    n_heads=8,
    e_layers=3,
    d_ff=256,
    patch_len=16,
    stride=8,
    dropout=0.2,
    target_dim=1
):
    """
    创建PatchTST模型

    推荐配置：
    - 短序列(<1000): patch_len=16, stride=8
    - 中序列(1000-3000): patch_len=32, stride=16
    - 长序列(>3000): patch_len=64, stride=32

    Args:
        input_dim: 输入特征数 (3: signal_0, signal_1, signal_2)
        seq_len: 输入序列长度
        pred_len: 预测长度
        d_model: Transformer维度 (推荐: 128-512)
        n_heads: 注意力头数 (推荐: 8-16)
        e_layers: Encoder层数 (推荐: 2-4)
        d_ff: FFN维度 (通常是d_model的2-4倍)
        patch_len: Patch长度
        stride: Patch步长
        dropout: Dropout率
        target_dim: 目标维度 (1表示只预测signal_1)
    """
    model = PatchTST(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        patch_len=patch_len,
        stride=stride,
        dropout=dropout,
        target_dim=target_dim
    )

    print(f"\n{'='*60}")
    print(f"PatchTST Model Created")
    print(f"{'='*60}")
    print(f"Input: [{input_dim} features] × {seq_len} steps")
    print(f"Output: {pred_len} steps")
    print(f"Patches: {(seq_len - patch_len) // stride + 1} (patch_len={patch_len}, stride={stride})")
    print(f"Transformer: d_model={d_model}, heads={n_heads}, layers={e_layers}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"{'='*60}\n")

    return model


if __name__ == '__main__':
    """测试PatchTST模型"""
    print("Testing PatchTST Model...")

    # 创建模型
    model = get_patchtst_model(
        input_dim=3,
        seq_len=2000,
        pred_len=1000,
        d_model=128,
        n_heads=8,
        e_layers=3,
        patch_len=32,
        stride=16
    )

    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 2000, 3)

    print(f"Input shape: {x.shape}")

    output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\n✅ PatchTST test passed!")
