"""
训练脚本
用于训练时间序列预测模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import json


class Trainer:
    """
    模型训练器

    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 设备 (cuda/cpu)
        learning_rate: 学习率
        save_dir: 模型保存目录
    """

    def __init__(self, model, train_loader, test_loader,
                 device='cuda', learning_rate=0.001, save_dir='./models', teacher_forcing_ratio=0.5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.teacher_forcing_ratio = teacher_forcing_ratio  # ✅ Teacher Forcing比例

        # 损失函数
        self.criterion = nn.MSELoss()

        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'learning_rate': []
        }

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets, last_values in pbar:  # ✅ 解包third value
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # last_values不需要在训练中使用，因为targets已经是差分值
            # last_values仅在评估时用于重建绝对值

            # 前向传播（传递 targets 以支持动态长度和Teacher Forcing）
            # ✅ 现在targets是差分值，模型学习预测差分
            self.optimizer.zero_grad()
            outputs = self.model(inputs, targets=targets, teacher_forcing_ratio=self.teacher_forcing_ratio)

            # 计算损失（基于差分值的MSE）
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets, last_values in tqdm(self.test_loader, desc='Evaluating'):  # ✅ 解包
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # last_values在这里也不用，因为我们只是计算差分值的MSE

                outputs = self.model(inputs, targets=targets)
                loss = self.criterion(outputs, targets)  # 基于差分值的损失

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, epochs=100, early_stopping_patience=10):
        """
        训练模型

        Args:
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值

        Returns:
            history: 训练历史
        """
        best_test_loss = float('inf')
        patience_counter = 0

        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            train_loss = self.train_epoch()

            # 评估
            test_loss = self.evaluate()

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['learning_rate'].append(current_lr)

            print(f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {current_lr:.6f}")

            # 学习率调度
            self.scheduler.step(test_loss)

            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, test_loss)
                print(f"✓ Saved best model (Test Loss: {test_loss:.6f})")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, test_loss)

        # 保存最终模型
        self.save_checkpoint('final_model.pth', epochs, test_loss)

        # 保存训练历史
        self.save_history()

        print("\nTraining completed!")
        print(f"Best Test Loss: {best_test_loss:.6f}")

        return self.history

    def save_checkpoint(self, filename, epoch, loss):
        """保存模型检查点"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }, filepath)

    def load_checkpoint(self, filename):
        """加载模型检查点"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['loss']

    def save_history(self):
        """保存训练历史到JSON文件"""
        filepath = os.path.join(self.save_dir, 'training_history.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {filepath}")


def calculate_metrics(predictions, targets):
    """
    计算评估指标

    Args:
        predictions: 预测值 [N, seq_len]
        targets: 真实值 [N, seq_len]

    Returns:
        metrics: 字典，包含各种指标
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # 计算R^2
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # MAPE (Mean Absolute Percentage Error)
    # 避免除零
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = float('inf')

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

    return metrics


def predict_batch(model, data_loader, device='cuda'):
    """
    批量预测

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备

    Returns:
        predictions: 所有预测结果（差分值）
        targets: 所有真实值（差分值）
        last_values: 用于重建的最后输入值
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_last_values = []

    with torch.no_grad():
        for inputs, targets, last_values in tqdm(data_loader, desc='Predicting'):  # ✅ 解包
            inputs = inputs.to(device)
            targets_device = targets.to(device)
            outputs = model(inputs, targets=targets_device)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_last_values.append(last_values.numpy())  # ✅ 保存last_values

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    last_values = np.concatenate(all_last_values, axis=0)

    return predictions, targets, last_values
