#!/usr/bin/env python3
"""
生成示例数据脚本
用于快速生成测试数据,方便测试代码

使用方法:
    python generate_sample_data.py --n_files 10 --n_samples 1000 --output_dir ./data
"""

import numpy as np
import pandas as pd
import os
import argparse


def generate_torque_curve(n_samples, seed=None):
    """
    生成模拟的扭矩曲线数据

    Args:
        n_samples: 样本数量
        seed: 随机种子

    Returns:
        time, torque, signal_0, signal_1, signal_2
    """
    if seed is not None:
        np.random.seed(seed)

    # 时间序列
    time = np.arange(0, n_samples * 0.01, 0.01)[:n_samples]

    # 基础频率和相位
    base_freq = np.random.uniform(0.3, 0.8)
    phase_shift = np.random.uniform(0, 2 * np.pi)

    # 生成扭矩曲线（模拟拧盖子的力学特征）
    # 特征1: 正弦波基础
    torque_base = 2 * np.sin(2 * np.pi * base_freq * time + phase_shift)

    # 特征2: 增长趋势（拧得越紧,力越大）
    trend = np.linspace(0, 1.5, n_samples)

    # 特征3: 阶跃变化（模拟螺纹的咬合）
    steps = np.zeros(n_samples)
    for i in range(3):
        step_pos = int(n_samples * (i + 1) / 4)
        steps[step_pos:] += 0.3

    # 组合扭矩信号
    torque = torque_base + trend + steps + np.random.normal(0, 0.15, n_samples)

    # Signal 0: 与扭矩相关但有延迟
    signal_0 = np.roll(torque, 5) * 0.8 + np.random.normal(0, 0.1, n_samples)

    # Signal 1: 主要信号,与扭矩强相关并有二次特征
    signal_1 = torque * 1.2 + 0.5 * torque ** 2 / 10 + np.random.normal(0, 0.12, n_samples)
    # 添加平滑的增长趋势
    signal_1 += np.linspace(0, 0.8, n_samples)

    # Signal 2: 较平稳的信号
    signal_2 = 1.5 * np.sin(2 * np.pi * base_freq * 0.5 * time) + \
               0.3 * trend + np.random.normal(0, 0.08, n_samples)

    return time, torque, signal_0, signal_1, signal_2


def generate_sample_data(n_files=10, n_samples=1000, output_dir='./data'):
    """
    生成多个示例CSV文件

    Args:
        n_files: 生成文件数量
        n_samples: 每个文件的样本数量
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("生成示例数据")
    print("=" * 60)
    print(f"文件数量: {n_files}")
    print(f"每个文件样本数: {n_samples}")
    print(f"输出目录: {output_dir}")
    print(f"时间长度: {n_samples * 0.01:.2f} 秒")
    print("=" * 60)

    for i in range(n_files):
        # 生成数据
        time, torque, signal_0, signal_1, signal_2 = generate_torque_curve(
            n_samples=n_samples,
            seed=42 + i  # 不同的种子生成不同的数据
        )

        # 创建DataFrame
        df = pd.DataFrame({
            'Time(s)': time,
            'Torque': torque,
            'signal_0': signal_0,
            'signal_1': signal_1,
            'signal_2': signal_2
        })

        # 保存文件
        filename = f'Data_{i // 2}_{i % 2}_open.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        print(f"✓ Generated: {filename} ({len(df)} samples)")

        # 显示统计信息
        if i == 0:
            print("\n第一个文件的统计信息:")
            print(df.describe())
            print()

    print("=" * 60)
    print(f"✓ 成功生成 {n_files} 个CSV文件!")
    print(f"文件保存在: {os.path.abspath(output_dir)}")
    print("=" * 60)


def visualize_sample(csv_file):
    """
    可视化单个CSV文件

    Args:
        csv_file: CSV文件路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装matplotlib才能可视化: pip install matplotlib")
        return

    # 读取数据
    df = pd.read_csv(csv_file)

    # 创建图表
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Torque
    axes[0].plot(df['Time(s)'], df['Torque'], linewidth=1.5, color='blue')
    axes[0].set_ylabel('Torque', fontsize=11)
    axes[0].set_title(f'Sample Data - {os.path.basename(csv_file)}', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Signal 0
    axes[1].plot(df['Time(s)'], df['signal_0'], linewidth=1.5, color='green')
    axes[1].set_ylabel('Signal 0', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Signal 1
    axes[2].plot(df['Time(s)'], df['signal_1'], linewidth=1.5, color='orange')
    axes[2].set_ylabel('Signal 1', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Signal 2
    axes[3].plot(df['Time(s)'], df['signal_2'], linewidth=1.5, color='red')
    axes[3].set_ylabel('Signal 2', fontsize=11)
    axes[3].set_xlabel('Time (s)', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(csv_file), 'sample_visualization.png'), dpi=150)
    print(f"可视化保存到: {os.path.join(os.path.dirname(csv_file), 'sample_visualization.png')}")
    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成示例扭矩曲线数据')

    parser.add_argument('--n_files', type=int, default=10,
                        help='生成文件数量 (默认: 10)')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='每个文件的样本数量 (默认: 1000)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='输出目录 (默认: ./data)')
    parser.add_argument('--visualize', action='store_true',
                        help='生成后可视化第一个文件')

    args = parser.parse_args()

    # 生成数据
    generate_sample_data(
        n_files=args.n_files,
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )

    # 可视化
    if args.visualize:
        first_file = os.path.join(args.output_dir, 'Data_0_0_open.csv')
        if os.path.exists(first_file):
            print("\n生成可视化...")
            visualize_sample(first_file)


if __name__ == '__main__':
    main()
