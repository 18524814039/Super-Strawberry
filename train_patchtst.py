"""
PatchTST训练脚本

使用PatchTST模型训练机械臂扭矩预测任务
结合Residual Learning方法预测差分值

使用方法:
    python train_patchtst.py
"""

import torch
import os
from src.data_loader import load_data_for_patchtst
from src.model import get_model
from src.train import Trainer
from src.evaluate import evaluate_model, evaluate_baseline

# ==================== 配置参数 ====================

# 数据参数
DATA_DIR = './data'
PATTERN = '*_open.csv'
TRAIN_SPLIT = 0.8

# 序列长度（PatchTST需要固定长度）
SEQ_LEN = 2000      # 输入序列长度
PRED_LEN = 1000     # 预测长度
STEP_SIZE = 500     # 滑动窗口步长（数据增强）

# 模型参数
INPUT_DIM = 3       # 3个信号 (signal_0, signal_1, signal_2)
D_MODEL = 128       # Transformer维度
N_HEADS = 8         # 注意力头数
E_LAYERS = 3        # Encoder层数
PATCH_LEN = 32      # Patch长度
STRIDE = 16         # Patch步长
DROPOUT = 0.2

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # PatchTST通常使用较小的学习率
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
TEACHER_FORCING_RATIO = 0.0  # PatchTST不使用Teacher Forcing

# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 保存目录
SAVE_DIR = './models_patchtst'
RESULTS_DIR = './results_patchtst'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 主程序 ====================

def main():
    print("\n" + "=" * 80)
    print("PatchTST 训练脚本")
    print("=" * 80)
    print(f"\n使用设备: {DEVICE}")
    print(f"数据目录: {DATA_DIR}")
    print(f"序列长度: 输入={SEQ_LEN}, 输出={PRED_LEN}")
    print(f"Patch配置: patch_len={PATCH_LEN}, stride={STRIDE}")
    print(f"模型配置: d_model={D_MODEL}, heads={N_HEADS}, layers={E_LAYERS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {EPOCHS} (早停patience={EARLY_STOPPING_PATIENCE})")

    # ==================== 1. 加载数据 ====================
    print("\n" + "=" * 80)
    print("步骤 1/5: 加载数据")
    print("=" * 80)

    train_loader, test_loader = load_data_for_patchtst(
        data_dir=DATA_DIR,
        pattern=PATTERN,
        train_split=TRAIN_SPLIT,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        signal_type='Fy',
        use_all_features=True,
        batch_size=BATCH_SIZE,
        step_size=STEP_SIZE
    )

    # ==================== 2. 创建模型 ====================
    print("\n" + "=" * 80)
    print("步骤 2/5: 创建PatchTST模型")
    print("=" * 80)

    model = get_model(
        model_type='patchtst',
        input_dim=INPUT_DIM,
        hidden_dim=D_MODEL,
        num_layers=E_LAYERS,
        output_length=PRED_LEN,
        dropout=DROPOUT,
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        n_heads=N_HEADS
    )

    # ==================== 3. 训练模型 ====================
    print("\n" + "=" * 80)
    print("步骤 3/5: 训练模型")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR,
        teacher_forcing_ratio=TEACHER_FORCING_RATIO
    )

    history = trainer.train(
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    # ==================== 4. 评估基线模型 ====================
    print("\n" + "=" * 80)
    print("步骤 4/5: 评估基线模型 (Persistence Baseline)")
    print("=" * 80)

    baseline_metrics = evaluate_baseline(test_loader, device=DEVICE)

    # ==================== 5. 评估PatchTST模型 ====================
    print("\n" + "=" * 80)
    print("步骤 5/5: 评估PatchTST模型")
    print("=" * 80)

    # 加载最佳模型
    trainer.load_checkpoint('best_model.pth')

    patchtst_metrics, predictions, targets, inputs = evaluate_model(
        model=trainer.model,
        data_loader=test_loader,
        device=DEVICE,
        save_dir=RESULTS_DIR
    )

    # ==================== 对比结果 ====================
    print("\n" + "=" * 80)
    print("最终对比结果")
    print("=" * 80)

    comparison = [
        ['指标', 'Baseline', 'PatchTST', '改进'],
        ['R²', f"{baseline_metrics['R2']:.4f}", f"{patchtst_metrics['R2']:.4f}",
         f"{((patchtst_metrics['R2'] - baseline_metrics['R2']) / abs(baseline_metrics['R2']) * 100):.1f}%"],
        ['RMSE', f"{baseline_metrics['RMSE']:.4f}", f"{patchtst_metrics['RMSE']:.4f}",
         f"{((baseline_metrics['RMSE'] - patchtst_metrics['RMSE']) / baseline_metrics['RMSE'] * 100):.1f}%"],
        ['MAE', f"{baseline_metrics['MAE']:.4f}", f"{patchtst_metrics['MAE']:.4f}",
         f"{((baseline_metrics['MAE'] - patchtst_metrics['MAE']) / baseline_metrics['MAE'] * 100):.1f}%"],
    ]

    # 打印表格
    col_widths = [max(len(str(row[i])) for row in comparison) + 2 for i in range(4)]

    for i, row in enumerate(comparison):
        print("  ".join(str(item).ljust(col_widths[j]) for j, item in enumerate(row)))
        if i == 0:
            print("  ".join("-" * col_widths[j] for j in range(4)))

    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"模型保存在: {SAVE_DIR}")
    print(f"结果保存在: {RESULTS_DIR}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
