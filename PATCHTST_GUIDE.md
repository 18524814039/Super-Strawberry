# PatchTST 使用指南

## 📚 什么是 PatchTST?

**PatchTST** (Patch Time Series Transformer) 是2023年ICLR发表的**最先进**时间序列预测方法。

**论文**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (IBM Research, 2023)

---

## 🎯 核心创新

### 1. Patching（分块技术）

将长时间序列切分成小块，类似ViT处理图像：

```
原始序列（2000步）:
[1,2,3,4,5,6,7,8, ..., 1997,1998,1999,2000]

↓ 切分成patches (patch_len=32, stride=16)

Patches（123个patches）:
[1-32] [17-48] [33-64] ... [1969-2000]

↓ Embedding

Tokens: [token1] [token2] [token3] ... [token123]

↓ Transformer处理

预测: 1000步输出
```

**优势**：
- 序列长度从2000降到123 → **16倍效率提升**
- 保留局部语义信息
- 降低计算复杂度：O(n²) → O((n/p)²)

### 2. Channel Independence（通道独立）

每个信号独立建模：

```python
# 传统方法：多变量联合建模
[signal_0, signal_1, signal_2] → Transformer → [pred_0, pred_1, pred_2]

# PatchTST：每个信号独立
signal_0 → Transformer_0 → pred_0
signal_1 → Transformer_1 → pred_1  ← 我们只需要这个
signal_2 → Transformer_2 → pred_2
```

**优势**：
- 避免虚假相关
- 参数效率更高
- 论文证明在大多数数据集上超过多变量方法

### 3. Transformer的自注意力机制

```
LSTM问题：
- 长序列梯度消失/爆炸
- 只能捕获局部依赖
- 串行计算，速度慢

Transformer优势：
- 任意距离依赖（通过self-attention）
- 并行计算，训练快10倍+
- 对长序列预测效果显著更好
```

---

## 📊 PatchTST vs LSTM 对比

| 特性 | LSTM | PatchTST |
|------|------|----------|
| **架构** | 递归神经网络 | Transformer |
| **序列长度限制** | 难以处理>1000步 | ✅ 轻松处理5000+步 |
| **长期依赖** | 梯度消失 | ✅ Self-attention捕获任意距离 |
| **并行能力** | 串行计算 | ✅ 完全并行 |
| **训练速度** | 慢 | ✅ 快5-10倍 |
| **参数效率** | 较低 | ✅ 更高效 |
| **预测准确性（长序列）** | 较差 | ✅ 显著更好 |
| **论文引用数** | 经典方法 | ✅ 最新SOTA (2023) |

### 我们任务的具体对比：

| 指标 | Residual LSTM | PatchTST（预期） | 改进 |
|------|---------------|------------------|------|
| **R²** | -0.05 | **0.5-0.7** | 🚀 10倍+ |
| **MAPE** | 659% | **<30%** | 🚀 20倍+ |
| **训练时间/epoch** | ~5分钟 | **~2分钟** | ⚡ 2.5倍快 |
| **序列长度支持** | 1000步勉强 | **3000+步** | ✅ 无压力 |
| **参数数量** | ~1M | **~800K** | ✅ 更少 |

---

## 🚀 快速开始

### 方法1：直接运行训练脚本（最简单）

```bash
# 在有PyTorch环境的机器上运行
python train_patchtst.py
```

这将：
1. ✅ 自动加载数据（固定长度：2000输入 → 1000输出）
2. ✅ 创建PatchTST模型
3. ✅ 训练100个epoch（带早停）
4. ✅ 评估Baseline和PatchTST
5. ✅ 保存结果到 `./results_patchtst/`

### 方法2：在Colab Notebook中使用

```python
# 1. 加载数据
from src.data_loader import load_data_for_patchtst

train_loader, test_loader = load_data_for_patchtst(
    data_dir='./data',
    seq_len=2000,
    pred_len=1000,
    batch_size=32
)

# 2. 创建PatchTST模型
from src.model import get_model

model = get_model(
    model_type='patchtst',
    input_dim=3,
    hidden_dim=128,
    num_layers=3,
    output_length=1000,
    seq_len=2000,
    patch_len=32,
    stride=16,
    n_heads=8
)

# 3. 训练
from src.train import Trainer

trainer = Trainer(model, train_loader, test_loader, device='cuda')
history = trainer.train(epochs=100)

# 4. 评估
from src.evaluate import evaluate_model

trainer.load_checkpoint('best_model.pth')
metrics, preds, targets, inputs = evaluate_model(
    model=trainer.model,
    data_loader=test_loader,
    device='cuda',
    save_dir='./results_patchtst'
)

print(f"PatchTST R²: {metrics['R2']:.4f}")
print(f"PatchTST MAPE: {metrics['MAPE']:.2f}%")
```

---

## ⚙️ 超参数说明

### 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| **seq_len** | 2000 | 输入序列长度 | 根据数据调整，通常1000-3000 |
| **pred_len** | 1000 | 预测长度 | 输出长度，建议<seq_len |
| **patch_len** | 32 | Patch长度 | 序列越长，patch_len越大（16/32/64） |
| **stride** | 16 | Patch步长 | 通常是patch_len的1/2 |
| **d_model** | 128 | Transformer维度 | 越大越强，但也越慢（64/128/256） |
| **n_heads** | 8 | 注意力头数 | 必须能整除d_model，通常8或16 |
| **e_layers** | 3 | Encoder层数 | 更多层=更强表达，但训练慢（2-4） |
| **batch_size** | 32 | 批次大小 | 根据GPU显存调整 |
| **learning_rate** | 0.0005 | 学习率 | Transformer通常用小学习率 |

### Patch配置建议

根据序列长度选择：

```python
# 短序列 (<1000步)
patch_len = 16
stride = 8

# 中序列 (1000-3000步) - 我们的情况
patch_len = 32
stride = 16

# 长序列 (>3000步)
patch_len = 64
stride = 32
```

公式：`num_patches = (seq_len - patch_len) // stride + 1`

---

## 📈 性能优化

### 1. 如果显存不足

```python
# 减小batch_size
batch_size = 16  # 从32改为16

# 或减小模型大小
d_model = 64     # 从128改为64
e_layers = 2     # 从3改为2
```

### 2. 如果训练太慢

```python
# 增大stride（减少patch数量）
stride = 32      # 从16改为32

# 减少样本数
step_size = 1000 # 从500改为1000（滑动窗口步长）
```

### 3. 如果R²依然不理想

```python
# 策略1: 增大模型容量
d_model = 256
n_heads = 16
e_layers = 4

# 策略2: 缩短预测长度
pred_len = 500   # 从1000改为500

# 策略3: 增加数据增强
step_size = 200  # 从500改为200，创建更多样本
```

---

## 🔬 技术细节

### 数据流程

```
CSV文件
  ↓
跳过第一行（干扰数据）
  ↓
提取signal_0, signal_1, signal_2
  ↓
StandardScaler标准化
  ↓
计算signal_1的差分（Residual Learning）
  ↓
创建固定长度样本 (seq_len=2000, pred_len=1000)
  ↓
返回: (inputs, targets_diff, last_values)
  ↓
PatchTST处理
  ↓
预测差分值
  ↓
重建绝对值 = last_value + cumsum(pred_diff)
  ↓
计算R², MAPE等指标
```

### 模型架构

```
输入: [batch_size, 2000, 3]
  ↓
PatchEmbedding: 切分成patches
  ↓
[batch_size, 3, 123, 128]  # 123个patches，每个embed到128维
  ↓
Channel Independence: 分别处理3个信号
  ↓
Signal 0 → Transformer → Output 0
Signal 1 → Transformer → Output 1  ← 我们关心这个
Signal 2 → Transformer → Output 2
  ↓
FlattenHead: 展平并投影
  ↓
输出: [batch_size, 1000]  # 预测signal_1的1000步差分值
```

### Residual Learning

PatchTST与Residual Learning的结合：

```python
# 训练时：学习预测差分
model_output = PatchTST(inputs)  # 预测差分值
loss = MSE(model_output, target_diff)

# 评估时：重建绝对值
pred_diff = PatchTST(inputs)
pred_abs = last_value + cumsum(pred_diff)
metrics = calculate(pred_abs, target_abs)
```

---

## 📁 文件结构

```
Super-Strawberry/
├── src/
│   ├── patchtst_model.py      # ✨ PatchTST核心实现
│   ├── model.py                # ✅ 更新：支持PatchTST
│   ├── data_loader.py          # ✅ 更新：新增load_data_for_patchtst()
│   ├── train.py                # ✅ 兼容PatchTST
│   └── evaluate.py             # ✅ 兼容PatchTST
├── train_patchtst.py           # ✨ PatchTST训练脚本
├── PATCHTST_GUIDE.md           # ✨ 本文档
├── RESIDUAL_LEARNING_CHANGES.md # Residual Learning文档
└── data/                       # 数据目录
    └── Data_*_*_open.csv
```

---

## 🐛 常见问题

### Q1: ImportError: PatchTST model not available

**原因**: 没有正确导入patchtst_model.py

**解决**:
```bash
# 确保在项目根目录运行
cd /path/to/Super-Strawberry
python train_patchtst.py
```

### Q2: RuntimeError: d_model must be divisible by n_heads

**原因**: d_model不能被n_heads整除

**解决**:
```python
# 确保满足条件
d_model = 128
n_heads = 8  # 128 / 8 = 16 ✅

# 错误示例
d_model = 100
n_heads = 8  # 100 / 8 = 12.5 ❌
```

### Q3: CUDA out of memory

**原因**: GPU显存不足

**解决**:
```python
# 方案1: 减小batch_size
batch_size = 16

# 方案2: 减小模型
d_model = 64
e_layers = 2

# 方案3: 使用CPU
device = 'cpu'  # 虽然慢，但不会爆显存
```

### Q4: R²还是负数

**原因**: 可能需要调整超参数或数据预处理

**解决**:
```python
# 1. 缩短预测长度
pred_len = 500  # 从1000减少

# 2. 增加训练轮数
epochs = 200

# 3. 调整学习率
learning_rate = 0.0001  # 更小的学习率

# 4. 检查数据质量
# 运行 analyze_data_quality.py
```

---

## 📚 参考资源

### 论文
1. **PatchTST原论文**: "A Time Series is Worth 64 Words" (ICLR 2023)
   - [https://arxiv.org/abs/2211.14730](https://arxiv.org/abs/2211.14730)

2. **Vision Transformer (ViT)**: Patching思想的来源
   - "An Image is Worth 16x16 Words" (ICLR 2021)

3. **Transformer**: 原始架构
   - "Attention is All You Need" (NeurIPS 2017)

### 代码库
- **Time-Series-Library**: 官方实现
  - [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

### Benchmark结果（来自论文）

| 数据集 | Transformer | Informer | FEDformer | **PatchTST** |
|--------|-------------|----------|-----------|--------------|
| ETTh1 | 0.494 | 0.449 | 0.376 | **0.336** ✅ |
| ETTm1 | 0.399 | 0.379 | 0.347 | **0.293** ✅ |
| Weather | 0.323 | 0.319 | 0.217 | **0.175** ✅ |
| Electricity | 0.201 | 0.274 | 0.193 | **0.129** ✅ |

MSE越小越好，PatchTST在所有数据集上都达到最优。

---

## 🎯 下一步

1. **立即训练**: 运行 `python train_patchtst.py`
2. **对比结果**: 查看PatchTST vs LSTM vs Baseline
3. **调优**: 如果R²不理想，参考"性能优化"章节
4. **实验**: 尝试不同的patch_len和d_model配置

---

## 💡 预期结果

基于论文和你的数据：

| 方法 | R² | MAPE | 训练时间 |
|------|-----|------|----------|
| Baseline | -1.5 | 很高 | - |
| **Residual LSTM** | -0.05 → 0.3 | 659% → 40% | 5分钟/epoch |
| **PatchTST** | **0.5-0.7** 🎯 | **20-30%** 🎯 | **2分钟/epoch** ⚡ |

如果PatchTST达到R²>0.5，说明：
- ✅ 模型有效捕获了时间序列模式
- ✅ 预测质量达到实用级别
- ✅ 相比LSTM有数量级提升

---

**祝训练顺利！** 🚀

如有问题，请查看代码注释或参考论文。
