"""
数据加载和预处理模块
用于加载机械臂扭矩CSV数据并进行时间序列预处理
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
import os


class TorqueDataset(Dataset):
    """
    机械臂扭矩时间序列数据集

    Args:
        csv_files: CSV文件路径列表
        input_length: 输入序列长度（例如100个时间步）
        output_length: 输出序列长度（例如400个时间步）
        signal_type: 要预测的信号类型 ('signal_0', 'signal_1', 'signal_2')
        use_all_features: 是否使用所有特征作为输入
        scaler: 数据标准化器
    """

    def __init__(self, csv_files, input_length=100, output_length=400,
                 signal_type='signal_1', use_all_features=True, scaler=None):
        self.input_length = input_length
        self.output_length = output_length
        self.signal_type = signal_type
        self.use_all_features = use_all_features
        self.scaler = scaler

        # 加载所有CSV文件
        self.data_list = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            self.data_list.append(df)

        # 创建序列样本
        self.samples = self._create_samples()

    def _create_samples(self):
        """从多个CSV文件中创建训练样本"""
        samples = []

        for df in self.data_list:
            # 确保数据长度足够
            total_length = self.input_length + self.output_length
            if len(df) < total_length:
                continue

            # 提取特征
            if self.use_all_features:
                # 使用所有特征：Time, signal_0, signal_1, signal_2
                features = df[['Time(s)', 'signal_0', 'signal_1', 'signal_2']].values
            else:
                # 仅使用目标信号
                features = df[[self.signal_type]].values

            # 提取目标信号（用于输出）
            target = df[self.signal_type].values

            # 在每个CSV文件中可以创建多个滑动窗口样本
            # 这里我们每隔50个时间步创建一个样本（可调整）
            step_size = 50
            for i in range(0, len(df) - total_length + 1, step_size):
                input_seq = features[i:i + self.input_length]
                output_seq = target[i + self.input_length:i + total_length]

                samples.append({
                    'input': input_seq,
                    'output': output_seq
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为tensor
        input_tensor = torch.FloatTensor(sample['input'])
        output_tensor = torch.FloatTensor(sample['output'])

        return input_tensor, output_tensor


def load_data(data_dir, pattern='*_open.csv', train_split=0.8,
              input_length=100, output_length=400,
              signal_type='signal_1', use_all_features=True,
              batch_size=32):
    """
    加载数据并创建训练集和测试集

    Args:
        data_dir: 数据目录路径
        pattern: CSV文件匹配模式
        train_split: 训练集比例
        input_length: 输入序列长度
        output_length: 输出序列长度
        signal_type: 要预测的信号类型
        use_all_features: 是否使用所有特征
        batch_size: 批次大小

    Returns:
        train_loader, test_loader, scaler
    """

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, pattern))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {data_dir} with pattern {pattern}")

    print(f"Found {len(csv_files)} CSV files")

    # 划分训练集和测试集
    np.random.seed(42)
    np.random.shuffle(csv_files)
    split_idx = int(len(csv_files) * train_split)
    train_files = csv_files[:split_idx]
    test_files = csv_files[split_idx:]

    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")

    # 创建数据集
    train_dataset = TorqueDataset(
        train_files,
        input_length=input_length,
        output_length=output_length,
        signal_type=signal_type,
        use_all_features=use_all_features
    )

    test_dataset = TorqueDataset(
        test_files,
        input_length=input_length,
        output_length=output_length,
        signal_type=signal_type,
        use_all_features=use_all_features
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Colab上设为0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


def get_sample_data_info(csv_file):
    """
    获取单个CSV文件的信息（用于调试）
    """
    df = pd.read_csv(csv_file)
    print(f"File: {csv_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())
    return df
