"""
测试residual learning实现

验证：
1. DataLoader正确返回3个值 (inputs, targets_diff, last_values)
2. 差分值可以正确重建为绝对值
3. 训练和评估流程不报错
"""

import torch
import numpy as np
from src.data_loader import load_data_dynamic
from src.model import get_model
from src.train import Trainer

def test_data_loader():
    """测试数据加载器是否正确返回3个值"""
    print("=" * 60)
    print("测试1: 数据加载器")
    print("=" * 60)

    try:
        train_loader, test_loader = load_data_dynamic(
            data_dir='./data',
            pattern='*_open.csv',
            train_split=0.8,
            input_ratio=2/3,
            output_ratio=1/3,
            signal_type='signal_1',
            use_all_features=True,
            batch_size=16,
            min_length=1000,
            bucket_size=200,
            step_size=100
        )

        # 获取一个batch
        for inputs, targets, last_values in train_loader:
            print(f"\n✅ 成功解包3个值:")
            print(f"   inputs shape: {inputs.shape}")
            print(f"   targets (差分值) shape: {targets.shape}")
            print(f"   last_values shape: {last_values.shape}")

            # 测试重建
            batch_size, output_length = targets.shape
            reconstructed = torch.zeros_like(targets)
            current = last_values.squeeze(-1)

            for t in range(output_length):
                current = current + targets[:, t]
                reconstructed[:, t] = current

            print(f"\n✅ 重建测试:")
            print(f"   最后一个输入值: {last_values[0].item():.4f}")
            print(f"   前5个差分值: {targets[0, :5].numpy()}")
            print(f"   前5个重建值: {reconstructed[0, :5].numpy()}")

            break

        print("\n✅ 数据加载器测试通过!")
        return True

    except Exception as e:
        print(f"\n❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 60)
    print("测试2: 模型前向传播")
    print("=" * 60)

    try:
        # 创建模型
        INPUT_DIM = 3  # signal_0, signal_1, signal_2
        OUTPUT_LENGTH = 500
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = get_model(
            model_type='lstm',
            input_dim=INPUT_DIM,
            hidden_dim=64,
            num_layers=2,
            output_length=OUTPUT_LENGTH,
            dropout=0.2
        ).to(DEVICE)

        # 创建假数据
        batch_size = 4
        input_length = 1000
        output_length = 500

        inputs = torch.randn(batch_size, input_length, INPUT_DIM).to(DEVICE)
        targets = torch.randn(batch_size, output_length).to(DEVICE)

        # 前向传播
        outputs = model(inputs, targets=targets, teacher_forcing_ratio=0.5)

        print(f"\n✅ 模型前向传播成功:")
        print(f"   输入 shape: {inputs.shape}")
        print(f"   目标 shape: {targets.shape}")
        print(f"   输出 shape: {outputs.shape}")
        print(f"   输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

        print("\n✅ 模型测试通过!")
        return True

    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """测试训练循环（只训练1个epoch）"""
    print("\n" + "=" * 60)
    print("测试3: 训练循环")
    print("=" * 60)

    try:
        # 加载数据
        train_loader, test_loader = load_data_dynamic(
            data_dir='./data',
            pattern='*_open.csv',
            train_split=0.8,
            input_ratio=2/3,
            output_ratio=1/3,
            signal_type='signal_1',
            use_all_features=True,
            batch_size=16,
            min_length=1000,
            bucket_size=200,
            step_size=100
        )

        # 创建模型
        INPUT_DIM = 3
        OUTPUT_LENGTH = 500
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = get_model(
            model_type='lstm',
            input_dim=INPUT_DIM,
            hidden_dim=64,
            num_layers=2,
            output_length=OUTPUT_LENGTH,
            dropout=0.2
        )

        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE,
            learning_rate=0.001,
            save_dir='./test_models',
            teacher_forcing_ratio=0.5
        )

        # 训练1个epoch
        print("\n开始训练1个epoch...")
        train_loss = trainer.train_epoch()
        test_loss = trainer.evaluate()

        print(f"\n✅ 训练循环测试通过!")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Test Loss: {test_loss:.6f}")

        return True

    except Exception as e:
        print(f"\n❌ 训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧪 Residual Learning 实现测试")
    print("=" * 60)
    print("\n关键改动:")
    print("1. 目标值改为差分 (target_diff = target[t+1] - target[t])")
    print("2. 模型学习预测差分，而不是绝对值")
    print("3. 评估时重建绝对值 (pred_abs = last_value + cumsum(pred_diff))")
    print("=" * 60 + "\n")

    # 运行测试
    results = []
    results.append(("数据加载器", test_data_loader()))
    results.append(("模型前向传播", test_model_forward()))
    results.append(("训练循环", test_training_loop()))

    # 总结
    print("\n\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有测试通过！可以开始完整训练了。")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")

    print("=" * 60)
