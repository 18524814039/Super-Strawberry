"""
评估和可视化模块
用于评估模型性能并可视化预测结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os


def evaluate_baseline(data_loader, device='cuda'):
    """
    评估Persistence Baseline（持久化基线）

    Baseline策略：用输入序列的最后一个值作为所有输出的预测
    这是最简单的baseline，如果LSTM连这个都打不过，说明模型完全失败

    Args:
        data_loader: 测试数据加载器
        device: 设备

    Returns:
        metrics: 评估指标字典
    """
    print("\n" + "=" * 60)
    print("📊 评估 Persistence Baseline")
    print("=" * 60)
    print("策略: 用输入序列最后一个 signal_1 值预测所有输出")
    print("-" * 60)

    all_predictions = []
    all_targets = []

    for inputs, targets in data_loader:
        # 提取 signal_1（假设是索引3，或者是第4列）
        # inputs shape: [batch_size, input_length, features]
        # 如果有5个特征(Time, Torque, signal_0, signal_1, signal_2)，signal_1是索引3

        # 获取输入序列的最后一个signal_1值
        last_signal1 = inputs[:, -1, 3]  # [batch_size]

        # 用这个值预测所有输出
        batch_size = inputs.size(0)
        output_length = targets.size(1)

        # Baseline预测：所有输出都是最后一个输入值
        baseline_pred = last_signal1.unsqueeze(1).expand(batch_size, output_length)

        all_predictions.append(baseline_pred.numpy())
        all_targets.append(targets.numpy())

    # 展平所有预测和目标
    predictions_flat = np.concatenate([pred.flatten() for pred in all_predictions])
    targets_flat = np.concatenate([tgt.flatten() for tgt in all_targets])

    # 计算指标
    mse = mean_squared_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    r2 = r2_score(targets_flat, predictions_flat)

    # MAPE
    mask = targets_flat != 0
    mape = np.mean(np.abs((targets_flat[mask] - predictions_flat[mask]) /
                          targets_flat[mask])) * 100 if mask.sum() > 0 else float('inf')

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

    # 打印指标
    print("\nBaseline Metrics:")
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.6f}")
    print("=" * 60)

    return metrics


def plot_training_history(history, save_path=None):
    """
    绘制训练历史

    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 学习率曲线
    axes[1].plot(history['learning_rate'], label='Learning Rate', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_predictions(inputs, predictions, targets, num_samples=4, save_path=None):
    """
    绘制预测结果对比

    Args:
        inputs: 输入序列 [N, input_len, features]
        predictions: 预测结果 [N, output_len]
        targets: 真实值 [N, output_len]
        num_samples: 显示样本数量
        save_path: 保存路径
    """
    num_samples = min(num_samples, len(predictions))

    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax = axes[i]

        # 输入序列长度
        input_len = inputs.shape[1]
        output_len = predictions.shape[1]

        # 时间轴
        input_time = np.arange(0, input_len)
        output_time = np.arange(input_len, input_len + output_len)

        # 绘制输入序列（使用signal_1，假设是最后一列或指定列）
        if inputs.shape[2] > 1:
            # 如果有多个特征，使用signal_1（假设是索引3）
            input_signal = inputs[i, :, -2] if inputs.shape[2] >= 4 else inputs[i, :, 0]
        else:
            input_signal = inputs[i, :, 0]

        ax.plot(input_time, input_signal, 'b-', label='Input Sequence', linewidth=2, alpha=0.7)

        # 绘制真实值
        ax.plot(output_time, targets[i], 'g-', label='Ground Truth', linewidth=2, alpha=0.7)

        # 绘制预测值
        ax.plot(output_time, predictions[i], 'r--', label='Prediction', linewidth=2, alpha=0.7)

        # 添加分界线
        ax.axvline(x=input_len, color='gray', linestyle='--', alpha=0.5, label='Prediction Start')

        # 计算误差
        mse = mean_squared_error(targets[i], predictions[i])
        mae = mean_absolute_error(targets[i], predictions[i])
        r2 = r2_score(targets[i], predictions[i])

        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Signal Value', fontsize=11)
        ax.set_title(f'Sample {i + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")

    plt.show()


def plot_error_distribution(predictions, targets, save_path=None):
    """
    绘制误差分布

    Args:
        predictions: 预测值 [N, seq_len]
        targets: 真实值 [N, seq_len]
        save_path: 保存路径
    """
    errors = predictions - targets

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 误差直方图
    axes[0, 0].hist(errors.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Prediction Error', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 误差散点图
    axes[0, 1].scatter(targets.flatten(), predictions.flatten(), alpha=0.3, s=1)
    axes[0, 1].plot([targets.min(), targets.max()],
                    [targets.min(), targets.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_xlabel('Ground Truth', fontsize=11)
    axes[0, 1].set_ylabel('Prediction', fontsize=11)
    axes[0, 1].set_title('Prediction vs Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 误差随时间变化
    time_errors = np.mean(np.abs(errors), axis=0)
    axes[1, 0].plot(time_errors, linewidth=2)
    axes[1, 0].set_xlabel('Time Step', fontsize=11)
    axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=11)
    axes[1, 0].set_title('MAE Over Time Steps', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 误差百分位数
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(np.abs(errors), p) for p in percentiles]
    axes[1, 1].bar(range(len(percentiles)), percentile_values, tick_label=percentiles)
    axes[1, 1].set_xlabel('Percentile', fontsize=11)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=11)
    axes[1, 1].set_title('Error Percentiles', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")

    plt.show()


def evaluate_model(model, data_loader, device='cuda', save_dir='./results'):
    """
    完整评估模型

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        save_dir: 结果保存目录

    Returns:
        metrics: 评估指标字典
        predictions: 预测结果
        targets: 真实值
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_predictions = []
    all_targets = []
    all_inputs = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs_device = inputs.to(device)
            targets_device = targets.to(device)
            outputs = model(inputs_device, targets=targets_device)

            all_inputs.append(inputs.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # ✅ 动态长度处理：将所有样本展平成一维数组用于计算指标
    # 而不是尝试concatenate不同长度的数组
    predictions_flat = np.concatenate([pred.flatten() for pred in all_predictions])
    targets_flat = np.concatenate([tgt.flatten() for tgt in all_targets])

    # 计算指标
    mse = mean_squared_error(targets_flat, predictions_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    r2 = r2_score(targets_flat, predictions_flat)

    # MAPE
    mask = targets_flat != 0
    mape = np.mean(np.abs((targets_flat[mask] - predictions_flat[mask]) /
                          targets_flat[mask])) * 100 if mask.sum() > 0 else float('inf')

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

    # 打印指标
    print("\n" + "=" * 50)
    print("Model Evaluation Metrics")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.6f}")
    print("=" * 50)

    # 可视化
    print("\nGenerating visualizations...")

    # 📊 从第一个batch中选择样本进行可视化（所有样本长度相同）
    sample_inputs = all_inputs[0]
    sample_predictions = all_predictions[0]
    sample_targets = all_targets[0]

    # 预测对比图
    plot_predictions(sample_inputs, sample_predictions, sample_targets, num_samples=min(4, len(sample_inputs)),
                    save_path=os.path.join(save_dir, 'predictions.png'))

    # 误差分布图（使用所有样本）
    # 找到所有batch中最小的序列长度
    min_seq_len = min(pred.shape[1] for pred in all_predictions)

    # 截断所有batch到最小长度，然后合并
    predictions_for_plot = []
    targets_for_plot = []
    for pred, tgt in zip(all_predictions, all_targets):
        predictions_for_plot.append(pred[:, :min_seq_len])
        targets_for_plot.append(tgt[:, :min_seq_len])

    predictions_for_plot = np.concatenate(predictions_for_plot, axis=0)
    targets_for_plot = np.concatenate(targets_for_plot, axis=0)

    plot_error_distribution(predictions_for_plot, targets_for_plot,
                           save_path=os.path.join(save_dir, 'error_distribution.png'))

    # 保存指标到文件
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    print(f"\nMetrics saved to {os.path.join(save_dir, 'metrics.csv')}")

    return metrics, all_predictions, all_targets, all_inputs


def predict_single_sequence(model, input_sequence, device='cuda'):
    """
    预测单个序列

    Args:
        model: 训练好的模型
        input_sequence: 输入序列 [input_len, features] 或 [1, input_len, features]
        device: 设备

    Returns:
        prediction: 预测结果 [output_len]
    """
    model.eval()

    # 确保维度正确
    if input_sequence.dim() == 2:
        input_sequence = input_sequence.unsqueeze(0)  # 添加batch维度

    with torch.no_grad():
        input_device = input_sequence.to(device)
        output = model(input_device)

    return output.cpu().numpy()[0]
