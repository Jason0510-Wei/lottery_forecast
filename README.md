# 双色球 & 大乐透彩票 AI 预测系统

> **现代化重构版本** - 基于 TensorFlow 2.x Keras API，提高代码可维护性和性能

## 项目简介

使用 LSTM 神经网络模型预测中国彩票（双色球和大乐透）开奖号码。该项目从网络爬取历史开奖数据，使用 LSTM 模型分别训练红球和蓝球的预测模型，最后进行下期号码预测。

> ⚠️ **免责声明**：此项目仅供娱乐学习，预测结果不保证准确性。彩票具有随机性，请理性购彩。

## 快速开始

### 1. 环境配置

#### 方式一：使用 venv（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：直接安装

```bash
pip install -r requirements.txt
```

### 2. 获取数据

爬取双色球历史数据：
```bash
python get_data.py --name ssq
```

或爬取大乐透历史数据：
```bash
python get_data.py --name dlt
```

**参数说明**：
- `--name`: 彩票类型，`ssq`(双色球) 或 `dlt`(大乐透)，默认为 `ssq`

**输出**：数据保存到 `data/ssq/data.csv` 或 `data/dlt/data.csv`

### 3. 训练模型

训练双色球模型：
```bash
python run_train_model.py --name ssq
```

或训练大乐透模型：
```bash
python run_train_model.py --name dlt
```

**参数说明**：
- `--name`: 彩票类型，默认为 `ssq`
- `--train_test_split`: 训练集占比，默认 0.7
- `--batch_size`: 批大小，默认 32
- `--epochs`: 训练轮数，默认 50

**输出**：
- 训练日志保存到 `logs/train_*.log`
- 模型保存到 `model/ssq/red_ball_model/` 和 `model/ssq/blue_ball_model/`

### 4. 进行预测

预测双色球下期号码：
```bash
python run_predict.py --name ssq
```

或预测大乐透下期号码：
```bash
python run_predict.py --name dlt
```

**输出**：
- 预测结果打印在控制台
- 预测日志保存到 `logs/predict_*.log`

## 项目结构

```
predict_Lottery_ticket/
├── config.py                 # 配置文件
├── get_data.py              # 数据爬取模块
├── modeling.py              # 模型定义
├── run_train_model.py       # 训练脚本
├── run_predict.py           # 预测脚本
├── requirements.txt         # 依赖列表
├── README.md                # 本文件
├── data/                    # 数据目录
│   ├── ssq/                # 双色球数据
│   └── dlt/                # 大乐透数据
├── model/                  # 模型目录
│   ├── ssq/                # 双色球模型
│   │   ├── red_ball_model/
│   │   └── blue_ball_model/
│   └── dlt/                # 大乐透模型
│       ├── red_ball_model/
│       └── blue_ball_model/
└── logs/                   # 日志目录
```

## 核心模块说明

### config.py
配置文件，包含：
- 彩票名称和数据路径
- 模型参数（embedding_size, hidden_size, epochs 等）
- 训练参数（learning_rate, validation_split 等）

### get_data.py
数据爬取模块：
- `get_current_number()`: 获取最新期号
- `spider()`: 爬取历史开奖数据
- `run()`: 主函数，爬取并保存数据

**数据源**：https://datachart.500.com/

### modeling.py
模型定义模块：

**LotteryPredictor**: 单个数字预测模型
- 基于 LSTM 的序列预测
- 支持多层 LSTM 和 Dropout
- 包含训练、预测、保存/加载等功能

**MultiOutputLSTM**: 多数字预测模型
- 用于预测多个球的号码
- 每个输出头独立预测一个球

### run_train_model.py
训练脚本：
- 数据加载和预处理
- 红球和蓝球模型分别训练
- 模型评估和保存

### run_predict.py
预测脚本：
- 加载训练好的模型
- 获取最新数据
- 预测下期号码
- 输出 Top K 概率预测

## 数据说明

### 双色球（SSQ）
- 红球：33 个数字（01-33），选 6 个
- 蓝球：16 个数字（01-16），选 1 个
- 数据字段：`红球_1, 红球_2, ..., 红球_6, 蓝球`

### 大乐透（DLT）
- 红球：35 个数字（01-35），选 5 个
- 蓝球：12 个数字（01-12），选 2 个
- 数据字段：`红球_1, 红球_2, ..., 红球_5, 蓝球_1, 蓝球_2`

## 模型架构

### 默认配置
- **嵌入层**：将整数编码转换为密集向量
- **LSTM 层**：捕捉时序特征（2 层，隐状态 128）
- **输出层**：softmax 分类输出

### 超参数
- **学习率**：0.001
- **批大小**：32
- **训练轮数**：50
- **Dropout**：0.2
- **早停耐心**：10

## 常见问题

### Q: 网络请求失败怎么办？
A: 检查网络连接和数据源是否可访问。如果官网数据源变更，请更新 `get_data.py` 中的 URL。

### Q: 模型训练很慢怎么办？
A: 
- 减少 `--epochs` 参数
- 增加 `--batch_size` 参数
- 减少数据量
- 使用 GPU 加速（需要 CUDA 环境）

### Q: 预测准确率很低怎么办？
A:
- 增加训练数据量
- 调整模型超参数
- 尝试更复杂的模型架构
- 收集更多历史数据

### Q: 如何使用 GPU 加速？
A: 安装 TensorFlow GPU 版本：
```bash
pip install tensorflow[and-cuda]
```

## 技术栈

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.9+ | 编程语言 |
| TensorFlow | 2.14.0+ | 深度学习框架 |
| NumPy | 1.24.3+ | 数值计算 |
| Pandas | 2.0.3+ | 数据处理 |
| scikit-learn | 1.3.0+ | 机器学习 |
| Loguru | 0.7.2+ | 日志记录 |

## 更新日志

### v2.0 (2025-11-12)
- ✅ 重构为现代 TensorFlow 2.x Keras API
- ✅ 改进数据预处理
- ✅ 优化模型架构
- ✅ 增强日志记录
- ✅ 改进错误处理
- ✅ 更新文档

### v1.0 (Original)
- 基于 TensorFlow 1.x with CRF
- LSTM + CRF 模型

## 许可证

MIT License

## 相关资源

- [TensorFlow 官网](https://www.tensorflow.org/)
- [Keras 文档](https://keras.io/)
- [彩票数据源](https://datachart.500.com/)

---

**最后更新**：2025 年 11 月 12 日


* 之前有issue反应，因为不同红球模型预测会有重复号码出现，所以将红球序列整体作为一个序列模型看待，推翻之前红球之间相互独立设定，
因为序列模型预测要引入crf层，相关API必须在 tf.compat.v1.disable_eager_execution()下，故整个模型采用 1.x 构建和训练模式，
在 2.x 的tensorflow中 tf.compat.v1.XXX 保留了 1.x 的接口方式。
