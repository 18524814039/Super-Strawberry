"""
深度学习模型定义
包含LSTM、GRU、Transformer和PatchTST等模型用于时间序列预测
"""

import torch
import torch.nn as nn
import math

# 导入PatchTST
try:
    from .patchtst_model import get_patchtst_model
    PATCHTST_AVAILABLE = True
except ImportError:
    PATCHTST_AVAILABLE = False
    print("Warning: PatchTST model not available")


class LSTMPredictor(nn.Module):
    """
    LSTM编码器-解码器模型
    用于序列到序列预测

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_layers: LSTM层数
        output_length: 输出序列长度
        dropout: Dropout比例
    """

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2,
                 output_length=400, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length

        # 编码器LSTM
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器LSTM
        self.decoder = nn.LSTM(
            1,  # 解码器输入维度（每次输入一个预测值）
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, target_len=None, targets=None, teacher_forcing_ratio=0.5):
        """
        前向传播（支持Teacher Forcing）

        Args:
            x: 输入序列 [batch_size, input_length, input_dim]
            target_len: 目标序列长度（显式指定，优先级最高）
            targets: 目标序列 [batch_size, target_length]（用于Teacher Forcing和自动推断长度）
            teacher_forcing_ratio: Teacher Forcing比例（训练时使用，0=纯自回归，1=纯teacher forcing）

        Returns:
            predictions: [batch_size, output_length]
        """
        batch_size = x.size(0)

        # 确定输出长度（优先级：target_len > targets.shape > self.output_length）
        if target_len is None:
            if targets is not None:
                target_len = targets.size(1)  # 自动适应目标长度（动态长度模式）
            else:
                target_len = self.output_length  # 使用默认长度

        # 编码器
        encoder_output, (hidden, cell) = self.encoder(x)

        # 解码器初始输入（使用编码器最后的输出）
        decoder_input = encoder_output[:, -1:, 0:1]  # [batch_size, 1, 1]

        # 存储所有预测
        predictions = []

        # ✅ 自回归解码（支持Teacher Forcing）
        for t in range(target_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.fc(decoder_output)  # [batch_size, 1, 1]
            predictions.append(prediction)

            # 决定下一个输入：Teacher Forcing vs 预测值
            if self.training and targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # ✅ 训练模式 + 有targets + 概率触发 → 使用真实值（Teacher Forcing）
                decoder_input = targets[:, t:t+1].unsqueeze(-1)  # [batch_size, 1, 1]
            else:
                # 推理模式 或 不使用Teacher Forcing → 使用预测值
                decoder_input = prediction

        # 拼接所有预测
        predictions = torch.cat(predictions, dim=1)  # [batch_size, target_len, 1]

        return predictions.squeeze(-1)  # [batch_size, target_len]


class GRUPredictor(nn.Module):
    """
    GRU编码器-解码器模型
    相比LSTM参数更少，训练更快
    """

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2,
                 output_length=400, dropout=0.2):
        super(GRUPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_length = output_length

        # 编码器GRU
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器GRU
        self.decoder = nn.GRU(
            1,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, target_len=None):
        batch_size = x.size(0)

        if target_len is None:
            target_len = self.output_length

        # 编码器
        encoder_output, hidden = self.encoder(x)

        # 解码器初始输入
        decoder_input = encoder_output[:, -1:, 0:1]

        predictions = []

        # 自回归解码
        for _ in range(target_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc(decoder_output)
            predictions.append(prediction)
            decoder_input = prediction

        predictions = torch.cat(predictions, dim=1)

        return predictions.squeeze(-1)


class TransformerPredictor(nn.Module):
    """
    Transformer模型用于时间序列预测
    使用注意力机制捕捉长距离依赖
    """

    def __init__(self, input_dim=4, d_model=128, nhead=8, num_layers=2,
                 output_length=400, dropout=0.2):
        super(TransformerPredictor, self).__init__()

        self.d_model = d_model
        self.output_length = output_length

        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.fc = nn.Linear(d_model, output_length)

    def forward(self, x, target_len=None):
        """
        Args:
            x: [batch_size, input_length, input_dim]

        Returns:
            predictions: [batch_size, output_length]
        """
        # 投影到d_model维度
        x = self.input_projection(x)  # [batch_size, input_length, d_model]

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        memory = self.transformer_encoder(x)  # [batch_size, input_length, d_model]

        # 使用最后一个时间步的输出
        output = memory[:, -1, :]  # [batch_size, d_model]

        # 直接预测整个输出序列
        predictions = self.fc(output)  # [batch_size, output_length]

        return predictions


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleLSTM(nn.Module):
    """
    简化版LSTM模型
    直接从输入序列预测输出序列，不使用解码器
    """

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2,
                 output_length=400, dropout=0.2):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_length)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_length, input_dim]

        Returns:
            predictions: [batch_size, output_length]
        """
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(x)

        # 使用最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        # 直接输出整个预测序列
        predictions = self.fc(last_hidden)  # [batch_size, output_length]

        return predictions


def get_model(model_type='lstm', input_dim=4, hidden_dim=128, num_layers=2,
              output_length=400, dropout=0.2, seq_len=None,
              patch_len=16, stride=8, n_heads=8):
    """
    根据类型获取模型

    Args:
        model_type: 模型类型 ('lstm', 'gru', 'transformer', 'simple_lstm', 'patchtst')
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度 (PatchTST中对应d_model)
        num_layers: 层数 (PatchTST中对应e_layers)
        output_length: 输出序列长度
        dropout: Dropout比例
        seq_len: 输入序列长度 (仅PatchTST需要)
        patch_len: Patch长度 (仅PatchTST)
        stride: Patch步长 (仅PatchTST)
        n_heads: 注意力头数 (仅PatchTST)

    Returns:
        model: 选择的模型
    """
    if model_type == 'lstm':
        return LSTMPredictor(input_dim, hidden_dim, num_layers, output_length, dropout)
    elif model_type == 'gru':
        return GRUPredictor(input_dim, hidden_dim, num_layers, output_length, dropout)
    elif model_type == 'transformer':
        return TransformerPredictor(input_dim, hidden_dim, 8, num_layers, output_length, dropout)
    elif model_type == 'simple_lstm':
        return SimpleLSTM(input_dim, hidden_dim, num_layers, output_length, dropout)
    elif model_type == 'patchtst':
        if not PATCHTST_AVAILABLE:
            raise ImportError("PatchTST model is not available. Check patchtst_model.py")
        if seq_len is None:
            raise ValueError("seq_len must be specified for PatchTST model")

        return get_patchtst_model(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=output_length,
            d_model=hidden_dim,
            n_heads=n_heads,
            e_layers=num_layers,
            d_ff=hidden_dim * 2,  # 通常是d_model的2倍
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
            target_dim=1  # 只预测signal_1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
