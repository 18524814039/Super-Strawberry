# 数据格式说明

## CSV文件格式要求

### 文件命名
- 格式: `Data_X_Y_open.csv`
- 示例: `Data_0_2_open.csv`, `Data_1_3_open.csv`
- "open"表示开盖数据

### CSV列格式

CSV文件必须包含以下5列（按顺序）:

| 列名 | 类型 | 说明 | 单位 |
|------|------|------|------|
| Time(s) | float | 时间戳 | 秒 |
| Torque | float | 扭矩值 | N·m |
| signal_0 | float | 信号0 | - |
| signal_1 | float | 信号1（主要预测目标） | - |
| signal_2 | float | 信号2 | - |

### 数据要求

1. **时间间隔**: 0.01秒 (100Hz采样率)
2. **数据长度**: 至少500个时间步（5秒数据）
   - 推荐: 1000个时间步或更多
3. **数据完整性**: 无缺失值
4. **数据格式**: CSV格式，逗号分隔

### CSV文件示例

```csv
Time(s),Torque,signal_0,signal_1,signal_2
0.00,0.123,0.456,0.789,0.012
0.01,0.125,0.458,0.791,0.013
0.02,0.127,0.460,0.793,0.014
0.03,0.129,0.462,0.795,0.015
...
```

### Python代码生成示例数据

如果你需要测试,可以使用以下代码生成示例数据:

```python
import numpy as np
import pandas as pd
import os

def generate_sample_data(n_samples=1000, save_dir='./data'):
    """生成示例数据用于测试"""
    os.makedirs(save_dir, exist_ok=True)

    # 生成5个示例文件
    for i in range(5):
        # 时间序列
        time = np.arange(0, n_samples * 0.01, 0.01)[:n_samples]

        # 生成类似扭矩曲线的数据
        # 使用正弦波 + 噪声 模拟真实曲线
        base_freq = 0.5 + i * 0.1

        torque = 2 * np.sin(2 * np.pi * base_freq * time) + np.random.normal(0, 0.1, n_samples)
        signal_0 = 1.5 * np.sin(2 * np.pi * base_freq * time + 0.5) + np.random.normal(0, 0.05, n_samples)
        signal_1 = 2.5 * np.sin(2 * np.pi * base_freq * time + 1.0) + np.random.normal(0, 0.08, n_samples)
        signal_2 = 1.8 * np.sin(2 * np.pi * base_freq * time + 1.5) + np.random.normal(0, 0.06, n_samples)

        # 添加趋势（模拟拧盖子的渐进过程）
        trend = np.linspace(0, 1, n_samples)
        signal_1 = signal_1 + trend * 0.5

        # 创建DataFrame
        df = pd.DataFrame({
            'Time(s)': time,
            'Torque': torque,
            'signal_0': signal_0,
            'signal_1': signal_1,
            'signal_2': signal_2
        })

        # 保存
        filename = f'Data_{i}_0_open.csv'
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Generated: {filepath}")

    print(f"\n✓ Generated 5 sample CSV files in {save_dir}")

# 运行
generate_sample_data()
```

## 数据目录结构

推荐的数据目录结构:

```
data/
├── Data_0_0_open.csv
├── Data_0_1_open.csv
├── Data_0_2_open.csv
├── Data_1_0_open.csv
├── Data_1_1_open.csv
├── ...
└── Data_N_M_open.csv
```

## 数据预处理

代码会自动进行以下预处理:

1. **滑动窗口采样**: 从每个CSV文件中提取多个训练样本
2. **训练/测试划分**: 按文件划分（默认80%训练，20%测试）
3. **批处理**: 自动组织成batch用于训练

## 常见问题

### Q1: 数据太少怎么办?
A:
- 使用数据增强: 添加噪声、缩放等
- 减小`step_size`以生成更多样本（在`data_loader.py`中修改）
- 收集更多数据

### Q2: 不同CSV文件长度不一样?
A: 没问题，代码会自动处理。只要每个文件长度 >= `input_length + output_length`

### Q3: 如何检查数据格式是否正确?
A: 运行以下代码:

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('your_file.csv')

# 检查列名
print("Columns:", df.columns.tolist())
# 应该输出: ['Time(s)', 'Torque', 'signal_0', 'signal_1', 'signal_2']

# 检查数据
print(df.head())
print(df.info())

# 检查时间间隔
time_diff = df['Time(s)'].diff().mean()
print(f"Average time interval: {time_diff:.4f} seconds")
# 应该约等于 0.0100
```

### Q4: 数据单位不对怎么办?
A: 数据会自动标准化，单位影响不大。但建议保持一致性。

## 数据质量检查清单

上传数据前请确认:

- [ ] CSV格式正确，包含5列
- [ ] 列名完全匹配（区分大小写）
- [ ] 时间间隔稳定在0.01秒
- [ ] 无缺失值（NaN）
- [ ] 数据长度足够（>=500步）
- [ ] 至少有5个CSV文件（推荐10+）
- [ ] 文件名符合规范（*_open.csv）

## 联系支持

如果数据格式有问题，请参考上述说明或查看示例生成代码。
