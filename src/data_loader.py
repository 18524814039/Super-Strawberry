"""
数据加载和预处理模块
用于加载机械臂扭矩CSV数据并进行时间序列预处理

支持两种模式：
1. 固定长度模式：所有样本使用相同的输入/输出长度
2. 动态比例模式：每个CSV按自身长度的比例划分（如前2/3预测后1/3）
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler
import glob
import os
from collections import defaultdict


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
                 signal_type='signal_1', use_all_features=True, scaler=None, step_size=50):
        self.input_length = input_length
        self.output_length = output_length
        self.signal_type = signal_type
        self.use_all_features = use_all_features
        self.scaler = scaler
        self.step_size = step_size

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
                # 使用所有特征：Time, Torque, signal_0, signal_1, signal_2
                features = df[['Time(s)', 'Torque', 'signal_0', 'signal_1', 'signal_2']].values
            else:
                # 仅使用目标信号
                features = df[[self.signal_type]].values

            # 提取目标信号（用于输出）
            target = df[self.signal_type].values

            # 在每个CSV文件中可以创建多个滑动窗口样本
            # step_size 可配置，默认50（可根据数据长度调整）
            for i in range(0, len(df) - total_length + 1, self.step_size):
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
              batch_size=32, step_size=50):
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
        use_all_features=use_all_features,
        step_size=step_size
    )

    test_dataset = TorqueDataset(
        test_files,
        input_length=input_length,
        output_length=output_length,
        signal_type=signal_type,
        use_all_features=use_all_features,
        step_size=step_size
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


# ==================== 动态长度支持 ====================

class DynamicTorqueDataset(Dataset):
    """
    动态长度机械臂扭矩时间序列数据集

    根据每个CSV的实际长度，按比例划分输入和输出序列
    例如：前2/3作为输入，后1/3作为输出

    Args:
        csv_files: CSV文件路径列表
        input_ratio: 输入序列占总长度的比例（例如 2/3）
        output_ratio: 输出序列占总长度的比例（例如 1/3）
        signal_type: 要预测的信号类型 ('signal_0', 'signal_1', 'signal_2')
        use_all_features: 是否使用所有特征作为输入
        min_length: 最小CSV长度要求（太短的CSV会被跳过）
        bucket_size: 长度分组的桶大小（相同桶内的样本可以batch在一起）
        step_size: 滑动窗口步长
    """

    def __init__(self, csv_files, input_ratio=2/3, output_ratio=1/3,
                 signal_type='signal_1', use_all_features=True,
                 min_length=1000, bucket_size=500, step_size=100):
        self.input_ratio = input_ratio
        self.output_ratio = output_ratio
        self.signal_type = signal_type
        self.use_all_features = use_all_features
        self.min_length = min_length
        self.bucket_size = bucket_size
        self.step_size = step_size

        # 加载所有CSV文件
        self.data_list = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if len(df) >= min_length:  # 只保留足够长的CSV
                self.data_list.append(df)

        print(f"Loaded {len(self.data_list)} CSV files (min_length >= {min_length})")

        # 创建序列样本（动态长度）
        self.samples = self._create_dynamic_samples()

        # 创建长度到样本索引的映射（用于分组batching）
        self._create_length_buckets()

    def _create_dynamic_samples(self):
        """从CSV文件中创建动态长度的训练样本"""
        samples = []
        length_stats = []

        for df in self.data_list:
            csv_length = len(df)

            # 根据比例计算输入和输出长度
            input_length = int(csv_length * self.input_ratio)
            output_length = int(csv_length * self.output_ratio)
            total_length = input_length + output_length

            # 确保总长度不超过CSV长度
            if total_length > csv_length:
                total_length = csv_length
                input_length = int(total_length * self.input_ratio)
                output_length = total_length - input_length

            length_stats.append(total_length)

            # 提取特征
            if self.use_all_features:
                features = df[['Time(s)', 'Torque', 'signal_0', 'signal_1', 'signal_2']].values
            else:
                features = df[[self.signal_type]].values

            # 提取目标信号
            target = df[self.signal_type].values

            # 使用滑动窗口创建多个样本
            for i in range(0, len(df) - total_length + 1, self.step_size):
                input_seq = features[i:i + input_length]
                output_seq = target[i + input_length:i + total_length]

                samples.append({
                    'input': input_seq,
                    'output': output_seq,
                    'length_bucket': (total_length // self.bucket_size) * self.bucket_size  # 长度桶标识
                })

        # 打印长度统计信息
        if length_stats:
            print(f"\n样本长度统计：")
            print(f"  最短: {min(length_stats)} 步")
            print(f"  最长: {max(length_stats)} 步")
            print(f"  平均: {np.mean(length_stats):.0f} 步")
            print(f"  中位数: {np.median(length_stats):.0f} 步")
            print(f"  总样本数: {len(samples)}")

        return samples

    def _create_length_buckets(self):
        """创建长度桶到样本索引的映射"""
        self.length_buckets = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            bucket = sample['length_bucket']
            self.length_buckets[bucket].append(idx)

        print(f"\n长度分组统计（bucket_size={self.bucket_size}）：")
        for bucket in sorted(self.length_buckets.keys()):
            count = len(self.length_buckets[bucket])
            print(f"  {bucket}-{bucket+self.bucket_size-1} 步: {count} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为tensor
        input_tensor = torch.FloatTensor(sample['input'])
        output_tensor = torch.FloatTensor(sample['output'])

        return input_tensor, output_tensor

    def get_length_bucket(self, idx):
        """获取样本的长度桶标识"""
        return self.samples[idx]['length_bucket']


class BucketBatchSampler(Sampler):
    """
    长度分组的批次采样器

    确保同一个batch内的样本来自相同的长度桶，
    这样它们的形状就是一致的，可以直接组成batch
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 按长度桶分组样本索引
        self.bucket_to_indices = dataset.length_buckets

    def __iter__(self):
        # 为每个桶创建batches
        all_batches = []

        for bucket, indices in self.bucket_to_indices.items():
            # 打乱桶内的样本顺序
            if self.shuffle:
                indices = indices.copy()
                np.random.shuffle(indices)

            # 将桶内样本分成多个batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # 只保留完整的batch
                    all_batches.append(batch)

        # 打乱所有batches的顺序
        if self.shuffle:
            np.random.shuffle(all_batches)

        # 返回batches
        for batch in all_batches:
            yield batch

    def __len__(self):
        # 计算总batch数
        total_batches = 0
        for indices in self.bucket_to_indices.values():
            total_batches += len(indices) // self.batch_size
        return total_batches


def load_data_dynamic(data_dir, pattern='*.csv', train_split=0.8,
                      input_ratio=2/3, output_ratio=1/3,
                      signal_type='signal_1', use_all_features=True,
                      batch_size=32, min_length=1000, bucket_size=500, step_size=100):
    """
    加载数据并创建训练集和测试集（动态长度模式）

    Args:
        data_dir: 数据目录路径
        pattern: CSV文件匹配模式
        train_split: 训练集比例
        input_ratio: 输入序列占比（例如 2/3）
        output_ratio: 输出序列占比（例如 1/3）
        signal_type: 要预测的信号类型
        use_all_features: 是否使用所有特征
        batch_size: 批次大小
        min_length: 最小CSV长度
        bucket_size: 长度分组的桶大小
        step_size: 滑动窗口步长

    Returns:
        train_loader, test_loader
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

    # 创建动态数据集
    print("\n创建训练集...")
    train_dataset = DynamicTorqueDataset(
        train_files,
        input_ratio=input_ratio,
        output_ratio=output_ratio,
        signal_type=signal_type,
        use_all_features=use_all_features,
        min_length=min_length,
        bucket_size=bucket_size,
        step_size=step_size
    )

    print("\n创建测试集...")
    test_dataset = DynamicTorqueDataset(
        test_files,
        input_ratio=input_ratio,
        output_ratio=output_ratio,
        signal_type=signal_type,
        use_all_features=use_all_features,
        min_length=min_length,
        bucket_size=bucket_size,
        step_size=step_size
    )

    # 创建批次采样器
    train_sampler = BucketBatchSampler(train_dataset, batch_size, shuffle=True)
    test_sampler = BucketBatchSampler(test_dataset, batch_size, shuffle=False)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=0
    )

    print(f"\n✅ 数据加载完成！")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    return train_loader, test_loader
