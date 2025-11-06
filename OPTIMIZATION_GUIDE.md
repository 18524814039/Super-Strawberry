# 模型优化指南

## 📊 当前表现分析

```
MSE       : 0.142129
RMSE      : 0.377000
MAE       : 0.266022
R²        : 0.949331 ✅ (很好)
MAPE      : 63.549221 ⚠️ (需要改进)
```

### 问题诊断
- **R² = 94.9%** 说明整体拟合度很好
- **MAPE = 63.5%** 说明在某些小值点上百分比误差较大
- 可能原因：模型在低信号值区域预测不准确

---

## 🚀 优化策略

### 1️⃣ **使用更强大的模型架构**

#### 推荐：Transformer 模型（注意力机制）

```python
# 训练 Transformer 模型
trainer = Trainer(
    model=get_model(
        model_type='transformer',  # 改用 Transformer
        input_dim=6,               # 使用所有特征
        hidden_dim=256,            # 增加隐藏维度
        num_layers=4,              # 增加层数
        output_length=400,
        dropout=0.3
    ),
    train_loader=train_loader,
    test_loader=test_loader,
    device=DEVICE,
    learning_rate=0.0005,          # 稍微降低学习率
    save_dir=MODEL_SAVE_DIR
)
```

#### 推荐：LSTM 编码器-解码器

```python
# 使用完整的 LSTM 编码器-解码器
model = get_model(
    model_type='lstm',             # LSTM 编码器-解码器
    input_dim=6,
    hidden_dim=256,                # 增加到 256
    num_layers=4,                  # 增加到 4 层
    output_length=400,
    dropout=0.3
)
```

---

### 2️⃣ **优化超参数配置**

#### 增加模型复杂度

```python
# 原配置
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.2

# 推荐配置（更强）
HIDDEN_DIM = 256        # 128 → 256
NUM_LAYERS = 4          # 3 → 4
DROPOUT = 0.3           # 0.2 → 0.3
```

#### 调整训练参数

```python
# 原配置
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# 推荐配置
LEARNING_RATE = 0.0005         # 降低学习率，更稳定
BATCH_SIZE = 64                # 增加批次大小
EPOCHS = 150                   # 更多轮次
EARLY_STOPPING_PATIENCE = 20   # 增加早停耐心
```

---

### 3️⃣ **改进损失函数**

当前使用的是 **MSE**，对小值预测不敏感。推荐使用**混合损失函数**：

#### 修改 `src/train.py`

```python
class Trainer:
    def __init__(self, ...):
        # 原代码: self.criterion = nn.MSELoss()

        # 改为混合损失：MSE + MAE
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

    def train_epoch(self):
        ...
        # 计算混合损失
        mse_loss = self.mse_criterion(outputs, targets)
        mae_loss = self.mae_criterion(outputs, targets)
        loss = 0.7 * mse_loss + 0.3 * mae_loss  # 权重可调整

        loss.backward()
        ...
```

#### 或者使用 Huber Loss（对异常值鲁棒）

```python
self.criterion = nn.HuberLoss(delta=1.0)
```

---

### 4️⃣ **数据预处理优化**

#### 检查数据分布

```python
import matplotlib.pyplot as plt

# 查看目标值分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(targets.flatten(), bins=50)
plt.title('Target Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(np.abs(targets.flatten())), bins=50)
plt.title('Log-scale Distribution')
plt.show()
```

#### 如果数据偏斜，考虑对数变换

```python
# 在 data_loader.py 中添加
import numpy as np

# 对目标值进行对数变换（如果值域跨度大）
targets_transformed = np.sign(targets) * np.log1p(np.abs(targets))

# 训练后需要反变换
predictions_original = np.sign(predictions) * (np.expm1(np.abs(predictions)))
```

---

### 5️⃣ **使用学习率预热和余弦退火**

#### 改进学习率调度策略

在 `src/train.py` 中修改：

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class Trainer:
    def __init__(self, ...):
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 使用余弦退火调度器（带重启）
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,      # 第一次重启的周期
            T_mult=2,    # 每次重启周期倍增
            eta_min=1e-6 # 最小学习率
        )
```

---

### 6️⃣ **增加正则化**

#### 添加 L2 正则化（权重衰减）

```python
# 在优化器中添加 weight_decay
self.optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5  # L2 正则化
)
```

#### 增加 Dropout

```python
# 创建模型时
model = get_model(
    model_type='lstm',
    input_dim=6,
    hidden_dim=256,
    num_layers=4,
    output_length=400,
    dropout=0.3  # 0.2 → 0.3
)
```

---

### 7️⃣ **集成学习（Ensemble）**

#### 训练多个模型并平均预测

```python
# 训练 3 个不同的模型
models = []
model_types = ['lstm', 'gru', 'transformer']

for model_type in model_types:
    model = get_model(
        model_type=model_type,
        input_dim=6,
        hidden_dim=256,
        num_layers=4,
        output_length=400
    )

    trainer = Trainer(model, train_loader, test_loader, device=DEVICE)
    trainer.train(epochs=100)
    models.append(trainer.model)

# 集成预测
def ensemble_predict(models, inputs):
    predictions = []
    for model in models:
        pred = model(inputs)
        predictions.append(pred)

    # 平均所有预测
    return torch.stack(predictions).mean(dim=0)
```

---

### 8️⃣ **数据增强**

#### 添加噪声增强

在 `src/data_loader.py` 中：

```python
class TorqueDataset(Dataset):
    def __init__(self, ..., add_noise=True):
        self.add_noise = add_noise

    def __getitem__(self, idx):
        inputs, target = ...

        # 训练时添加小量噪声
        if self.add_noise:
            noise = torch.randn_like(inputs) * 0.01
            inputs = inputs + noise

        return inputs, target
```

---

## 🎯 推荐的快速优化方案

### 方案 A：最快见效（5分钟）

```python
# 1. 使用 Transformer
model = get_model('transformer', input_dim=6, hidden_dim=256, num_layers=4)

# 2. 降低学习率
trainer = Trainer(model, train_loader, test_loader, learning_rate=0.0005)

# 3. 训练更多轮次
history = trainer.train(epochs=150, early_stopping_patience=20)
```

### 方案 B：中等优化（15分钟）

```python
# 1. 增加模型复杂度
model = get_model(
    model_type='lstm',
    input_dim=6,
    hidden_dim=256,  # 增大
    num_layers=4,    # 增多
    dropout=0.3
)

# 2. 修改损失函数为 Huber Loss
# 在 train.py 中: self.criterion = nn.HuberLoss(delta=1.0)

# 3. 使用余弦退火学习率
# 在 train.py 中添加 CosineAnnealingWarmRestarts

# 4. 训练
trainer = Trainer(model, train_loader, test_loader, learning_rate=0.001)
history = trainer.train(epochs=150)
```

### 方案 C：最佳效果（30分钟）

```python
# 1. 训练多个模型
model_configs = [
    {'type': 'lstm', 'hidden': 256, 'layers': 4},
    {'type': 'gru', 'hidden': 256, 'layers': 4},
    {'type': 'transformer', 'hidden': 256, 'layers': 4}
]

models = []
for config in model_configs:
    model = get_model(
        model_type=config['type'],
        input_dim=6,
        hidden_dim=config['hidden'],
        num_layers=config['layers'],
        dropout=0.3
    )

    trainer = Trainer(model, train_loader, test_loader, learning_rate=0.0005)
    trainer.train(epochs=150, early_stopping_patience=20)
    models.append(trainer.model)

# 2. 集成预测
predictions_ensemble = []
for model in models:
    pred = model(test_inputs)
    predictions_ensemble.append(pred)

final_predictions = torch.stack(predictions_ensemble).mean(dim=0)
```

---

## 📈 预期改进效果

| 优化方案 | 预期 R² | 预期 MAPE | 训练时间 |
|---------|---------|-----------|----------|
| 当前     | 0.949   | 63.5%     | -        |
| 方案 A   | 0.960+  | 45-50%    | +5 分钟  |
| 方案 B   | 0.965+  | 35-40%    | +15 分钟 |
| 方案 C   | 0.970+  | 25-30%    | +30 分钟 |

---

## 🔍 调试建议

### 查看预测曲线

```python
# 绘制更多样本查看效果
plot_predictions(inputs, predictions, targets, num_samples=8)
```

### 分析误差分布

```python
from src.evaluate import plot_error_distribution
plot_error_distribution(predictions, targets)
```

### 检查损失曲线是否收敛

```python
plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Test')
plt.legend()
plt.show()
```

---

## 💡 终极建议

1. **先尝试方案 A**（最快）- 使用 Transformer + 降低学习率
2. **如果还不够好**，尝试方案 B - 混合损失 + 余弦退火
3. **追求极致性能**，使用方案 C - 集成学习

**关键点**：
- ✅ 使用所有 6 个特征（不只是 signal_1）
- ✅ 增加模型容量（hidden_dim=256, num_layers=4）
- ✅ 降低学习率提高稳定性
- ✅ 使用更好的学习率调度策略
- ✅ 考虑集成多个模型

祝训练顺利！🚀
