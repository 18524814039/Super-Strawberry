# 机械臂扭矩曲线预测 - Super Strawberry

基于深度学习的机械臂拧盖子力学曲线预测系统

## 项目简介

本项目使用深度学习技术预测机械臂拧瓶盖时的力学曲线。通过输入前面一小段序列数据（例如前100个时间步），模型能够预测后续完整的力学曲线（例如后400个时间步）。

### 任务描述

- **输入**: 前N个时间步的多维特征数据 (Time, signal_0, signal_1, signal_2)
- **输出**: 后M个时间步的目标信号预测值
- **数据频率**: 0.01秒/步 (100Hz采样率)
- **示例**: 输入前1秒数据 → 预测后4秒的力学曲线

## 项目结构

```
Super-Strawberry/
├── src/
│   ├── data_loader.py      # 数据加载和预处理模块
│   ├── model.py            # 深度学习模型定义 (LSTM, GRU, Transformer)
│   ├── train.py            # 训练脚本和训练器类
│   └── evaluate.py         # 评估和可视化模块
├── notebooks/
│   └── train_colab.ipynb   # Google Colab训练notebook (主要使用)
├── data/                   # 数据目录 (需自行添加CSV文件)
├── models/                 # 保存训练好的模型
├── results/                # 评估结果和可视化图表
├── requirements.txt        # Python依赖
└── README.md              # 项目说明文档
```

## 快速开始

### 方法1: 在Google Colab上运行 (推荐)

1. **打开Colab Notebook**
   ```bash
   # 在GitHub上打开 notebooks/train_colab.ipynb
   # 或者上传到你的Google Drive
   ```

2. **准备数据**
   - 将CSV文件上传到Colab或挂载Google Drive
   - CSV格式要求: `Time(s), signal_0, signal_1, signal_2`
   - 文件命名: `Data_*_*_open.csv`

3. **运行Notebook**
   - 按顺序执行所有单元格
   - 根据需要调整配置参数
   - 等待训练完成并查看结果

### 方法2: 本地运行

1. **克隆仓库**
   ```bash
   git clone https://github.com/your-repo/Super-Strawberry.git
   cd Super-Strawberry
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**
   ```bash
   # 将CSV文件放在 data/ 目录下
   mkdir -p data
   # 复制你的 *_open.csv 文件到 data/
   ```

4. **使用Python脚本训练**
   ```python
   from src.data_loader import load_data
   from src.model import get_model
   from src.train import Trainer

   # 加载数据
   train_loader, test_loader = load_data(
       data_dir='./data',
       input_length=100,
       output_length=400,
       signal_type='signal_1'
   )

   # 创建模型
   model = get_model(
       model_type='simple_lstm',
       input_dim=4,
       hidden_dim=128,
       output_length=400
   )

   # 训练
   trainer = Trainer(model, train_loader, test_loader)
   history = trainer.train(epochs=100)
   ```

## 配置参数说明

### 数据参数
- `INPUT_LENGTH`: 输入序列长度 (默认100)
- `OUTPUT_LENGTH`: 输出序列长度 (默认400)
- `SIGNAL_TYPE`: 要预测的信号 ('signal_0', 'signal_1', 'signal_2')
- `USE_ALL_FEATURES`: 是否使用所有特征作为输入 (默认True)

### 模型参数
- `MODEL_TYPE`: 模型类型
  - `'simple_lstm'`: 简化LSTM (推荐,训练快)
  - `'lstm'`: LSTM编码器-解码器
  - `'gru'`: GRU编码器-解码器
  - `'transformer'`: Transformer模型
- `HIDDEN_DIM`: 隐藏层维度 (默认128)
- `NUM_LAYERS`: 网络层数 (默认3)
- `DROPOUT`: Dropout比例 (默认0.2)

### 训练参数
- `BATCH_SIZE`: 批次大小 (默认32)
- `LEARNING_RATE`: 学习率 (默认0.001)
- `EPOCHS`: 最大训练轮数 (默认100)
- `EARLY_STOPPING_PATIENCE`: 早停耐心值 (默认15)

## 模型架构

### 1. SimpleLSTM (推荐)
- 最简单快速的模型
- 使用LSTM编码输入序列
- 通过全连接层直接输出完整预测序列

### 2. LSTM Encoder-Decoder
- 编码器-解码器架构
- 自回归生成预测序列
- 适合长序列预测

### 3. GRU Encoder-Decoder
- 类似LSTM但参数更少
- 训练速度更快
- 效果与LSTM相近

### 4. Transformer
- 基于注意力机制
- 能捕捉长距离依赖
- 参数较多,需要更多数据

## 评估指标

模型评估使用以下指标:
- **MSE** (Mean Squared Error): 均方误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **MAE** (Mean Absolute Error): 平均绝对误差
- **R²** (R-squared): 决定系数
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差

## 使用示例

### 训练signal_1
```python
# 在notebook中设置
SIGNAL_TYPE = 'signal_1'
MODEL_TYPE = 'simple_lstm'
```

### 训练signal_0和signal_2
```python
# 效果好后,改变SIGNAL_TYPE重新训练
SIGNAL_TYPE = 'signal_0'  # 或 'signal_2'
```

### 预测新数据
```python
from src.evaluate import predict_single_sequence

# 加载模型
trainer.load_checkpoint('best_model.pth')

# 预测
prediction = predict_single_sequence(
    model=trainer.model,
    input_sequence=your_input_data,
    device='cuda'
)
```

## 结果可视化

训练完成后会自动生成:
- `training_history.png`: 训练损失和学习率曲线
- `predictions.png`: 预测结果对比图
- `error_distribution.png`: 误差分布分析
- `metrics.csv`: 详细评估指标

## 常见问题

### Q1: 如何选择输入/输出长度?
A: 根据实际需求调整。例如:
- 拧30度预测180度: 根据采样率计算对应的时间步数
- 时间步 = 角度 / 角速度 / 采样间隔

### Q2: 训练时间过长?
A: 尝试:
- 减少`EPOCHS`
- 增加`BATCH_SIZE`
- 使用`simple_lstm`模型
- 确保使用GPU

### Q3: 预测效果不好?
A: 可以:
- 增加`HIDDEN_DIM`
- 增加`NUM_LAYERS`
- 减少`DROPOUT`
- 收集更多训练数据
- 尝试不同模型类型

### Q4: 内存不足?
A: 减小:
- `BATCH_SIZE`
- `HIDDEN_DIM`
- `NUM_LAYERS`

## 技术栈

- **PyTorch**: 深度学习框架
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Matplotlib/Seaborn**: 可视化
- **scikit-learn**: 评估指标

## 后续改进方向

1. **多任务学习**: 同时预测signal_0, signal_1, signal_2
2. **注意力机制**: 添加注意力层提升性能
3. **集成学习**: 组合多个模型的预测结果
4. **在线学习**: 实时更新模型
5. **异常检测**: 识别异常力学曲线

## 许可证

MIT License

## 联系方式

如有问题或建议,请提Issue或Pull Request。

---

**开始训练**: 打开 `notebooks/train_colab.ipynb` 开始你的第一次训练!
