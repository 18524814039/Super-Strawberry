#!/usr/bin/env python3
"""
快速训练脚本 - 命令行版本
用于快速开始训练而无需打开notebook

使用方法:
    python quick_train.py --data_dir ./data --signal signal_1 --epochs 100

参数说明:
    --data_dir: 数据目录路径
    --signal: 要预测的信号 (signal_0, signal_1, signal_2)
    --input_len: 输入序列长度
    --output_len: 输出序列长度
    --model: 模型类型 (simple_lstm, lstm, gru, transformer)
    --hidden_dim: 隐藏层维度
    --num_layers: 网络层数
    --batch_size: 批次大小
    --epochs: 训练轮数
    --lr: 学习率
"""

import argparse
import os
import sys
import torch

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from model import get_model
from train import Trainer
from evaluate import evaluate_model, plot_training_history


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='机械臂扭矩曲线预测 - 快速训练')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录路径')
    parser.add_argument('--signal', type=str, default='signal_1',
                        choices=['signal_0', 'signal_1', 'signal_2'],
                        help='要预测的信号')
    parser.add_argument('--input_len', type=int, default=100,
                        help='输入序列长度')
    parser.add_argument('--output_len', type=int, default=400,
                        help='输出序列长度')

    # 模型参数
    parser.add_argument('--model', type=str, default='simple_lstm',
                        choices=['simple_lstm', 'lstm', 'gru', 'transformer'],
                        help='模型类型')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='网络层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout比例')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--early_stop', type=int, default=15,
                        help='早停耐心值')

    # 其他参数
    parser.add_argument('--use_all_features', action='store_true', default=True,
                        help='使用所有特征作为输入')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='计算设备')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("机械臂扭矩曲线预测 - 快速训练")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  预测信号: {args.signal}")
    print(f"  输入长度: {args.input_len}")
    print(f"  输出长度: {args.output_len}")
    print(f"  模型类型: {args.model}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  网络层数: {args.num_layers}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  计算设备: {device}")
    print("=" * 60)

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"\n错误: 数据目录 '{args.data_dir}' 不存在!")
        print("请创建数据目录并添加CSV文件")
        return

    # 加载数据
    print("\n加载数据...")
    try:
        train_loader, test_loader = load_data(
            data_dir=args.data_dir,
            pattern='*_open.csv',
            train_split=args.train_split,
            input_length=args.input_len,
            output_length=args.output_len,
            signal_type=args.signal,
            use_all_features=args.use_all_features,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"\n错误: 加载数据失败!")
        print(f"详细信息: {e}")
        return

    # 创建模型
    print("\n创建模型...")
    input_dim = 5 if args.use_all_features else 1
    model = get_model(
        model_type=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_length=args.output_len,
        dropout=args.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # 创建训练器
    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )

    # 开始训练
    print("\n开始训练...")
    print("=" * 60)
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stop
    )

    # 可视化训练历史
    print("\n生成训练历史图表...")
    os.makedirs(args.results_dir, exist_ok=True)
    plot_training_history(
        history,
        save_path=os.path.join(args.results_dir, f'training_history_{args.signal}.png')
    )

    # 评估模型
    print("\n评估模型...")
    trainer.load_checkpoint('best_model.pth')
    metrics, predictions, targets, inputs = evaluate_model(
        model=trainer.model,
        data_loader=test_loader,
        device=device,
        save_dir=os.path.join(args.results_dir, args.signal)
    )

    # 总结
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"模型保存在: {args.save_dir}")
    print(f"结果保存在: {args.results_dir}")
    print("\n最终评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
