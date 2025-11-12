# Residual Learning 实现说明

## 核心改进

基于2024年最新论文研究，实现了**残差学习（Residual Learning）**方法来预测时间序列。

### 原理

**之前**: 模型直接预测绝对值
```
输入: [100, 105, 103, ...]  →  输出: [110, 115, 112, ...]
```

**现在**: 模型预测差分（变化量）
```
输入: [100, 105, 103, ...]
       ↓
差分: [5, -2, 7, ...]        ←  模型学习这个
       ↓
重建: 103 + 5 = 108
      108 + (-2) = 106
      106 + 7 = 113
      ...
```

### 为什么有效？

1. **简化任务**: 预测"变化了多少"比预测"绝对值是多少"更容易
2. **减少方差**: 差分值的范围通常比绝对值小，模型更容易学习
3. **论文支持**:
   - "PSO-LSTM for robotic torque prediction" (2024)
   - "Residual LSTM for signal prediction" (Nature 2024)

---

## 代码改动

### 1. data_loader.py

**位置**: `_create_dynamic_samples()` 方法（第305-344行）

**改动**:
```python
# 计算差分
target_diff = np.diff(target, prepend=target[0])

# 标准化差分值（而不是原始值）
target_diff = (target_diff - np.mean(target_diff)) / (np.std(target_diff) + 1e-8)

# 保存最后一个输入值（用于重建）
last_input_value = features[i + input_length - 1, signal_idx]

# 样本包含差分值和重建所需的last_value
samples.append({
    'input': input_seq,
    'output': output_seq,  # 差分值
    'last_input_value': last_input_value
})
```

**返回值变化**:
- 之前: `return inputs, targets`
- 现在: `return inputs, targets_diff, last_values`

---

### 2. train.py

**位置**: `train_epoch()`, `evaluate()`, `predict_batch()`

**改动**:
```python
# 训练时解包3个值
for inputs, targets, last_values in train_loader:
    # targets现在是差分值
    outputs = model(inputs, targets=targets)  # 模型学习预测差分
    loss = criterion(outputs, targets)  # 基于差分的MSE
```

**注意**: 训练时不需要使用`last_values`，因为损失函数是基于差分值计算的。

---

### 3. evaluate.py

**位置**: `evaluate_baseline()`, `evaluate_model()`

**改动**: 重建绝对值用于评估

```python
# 重建绝对值预测
predictions_abs = torch.zeros_like(outputs)
current_value = last_values.squeeze(-1)  # 起点

for t in range(output_length):
    current_value = current_value + outputs[:, t]  # 累积加差分
    predictions_abs[:, t] = current_value

# 对真实值做同样的重建
targets_abs = torch.zeros_like(targets)
current_target = last_values.squeeze(-1)

for t in range(output_length):
    current_target = current_target + targets[:, t]
    targets_abs[:, t] = current_target

# 计算指标时使用绝对值
metrics = calculate_metrics(predictions_abs, targets_abs)
```

---

## 使用方法

### 测试实现

```bash
python test_residual_implementation.py
```

这将验证:
1. DataLoader正确返回3个值
2. 差分可以正确重建为绝对值
3. 训练和评估流程正常运行

### 完整训练

使用更新后的notebook:
```bash
# 在Colab中运行 training_optimized.ipynb
```

或者使用Python脚本:
```python
from src.data_loader import load_data_dynamic
from src.model import get_model
from src.train import Trainer

# 加载数据（自动使用residual learning）
train_loader, test_loader = load_data_dynamic(
    data_dir='./data',
    pattern='*_open.csv',
    input_ratio=2/3,
    output_ratio=1/3,
    use_all_features=True,  # INPUT_DIM=3
    batch_size=512
)

# 训练模型
model = get_model('lstm', input_dim=3, hidden_dim=128, num_layers=2)
trainer = Trainer(model, train_loader, test_loader)
history = trainer.train(epochs=100)
```

---

## 预期改进

基于之前结果:
- **Baseline R²**: -1.5
- **LSTM R² (旧方法)**: -0.05

使用残差学习后，预期:
- **LSTM R² (新方法)**: > 0.3 (保守估计)
- **MAPE**: < 40% (保守估计)

如果R²依然为负，可能需要:
1. 缩短预测长度（从1000-3000步缩短到200-500步）
2. 尝试CNN-LSTM混合架构
3. 调整超参数（hidden_dim, num_layers, learning_rate）

---

## 向后兼容性

**重要**: 旧的notebook和脚本需要更新以适配新的3值返回格式。

已更新的文件:
- ✅ `src/data_loader.py`
- ✅ `src/train.py`
- ✅ `src/evaluate.py`
- ⚠️ `training_optimized.ipynb` (需要手动更新或重新生成)

未更新的文件（如有使用）:
- `src/model.py` (无需改动)
- 其他自定义脚本

---

## 调试技巧

如果遇到错误:

1. **ValueError: too many values to unpack**
   - 原因: 旧代码只期待2个返回值
   - 修复: `for inputs, targets in loader:` → `for inputs, targets, last_values in loader:`

2. **维度不匹配**
   - 检查: `last_values.shape` 应该是 `[batch_size, 1]`
   - 检查: `targets.shape` 应该是 `[batch_size, output_length]`

3. **R²依然为负**
   - 尝试: 减小OUTPUT_RATIO (例如从1/3改为1/5)
   - 尝试: 增大模型容量 (hidden_dim=256, num_layers=3)
   - 检查: 数据是否有噪声或异常值

---

## 参考文献

1. Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"
2. He et al. (2016) - "Deep Residual Learning for Image Recognition"
3. Recent papers on time series forecasting with residual connections (2024)

---

## 联系方式

如有问题，请查看:
- GitHub Issues
- training_optimized.ipynb 中的注释
- test_residual_implementation.py 的测试用例
