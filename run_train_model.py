# -*- coding:utf-8 -*-
"""
训练脚本 - 使用现代 TensorFlow 2.x
Updated Version
"""
import argparse
import os
import numpy as np
import pandas as pd
from loguru import logger
import tensorflow as tf

from config import name_path, model_args, model_path, model_names, data_file_name
from modeling import LotteryPredictor, MultiOutputLSTM, MultiLabelLSTM

# 配置日志
logger.add("logs/train_{time}.log", rotation="500 MB", retention="7 days")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球(ssq)/大乐透(dlt)")
parser.add_argument('--train_test_split', default=0.7, type=float, help="训练集占比 (> 0.5)")
parser.add_argument('--batch_size', default=32, type=int, help="批大小")
parser.add_argument('--epochs', default=50, type=int, help="训练轮数")
args = parser.parse_args()


def load_data(lottery_name):
    """加载数据"""
    data_path = os.path.join(name_path[lottery_name]["path"], data_file_name)
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("请先运行: python get_data.py --name {}".format(lottery_name))
        return None
    
    data = pd.read_csv(data_path)
    logger.info(f"数据已加载，共 {len(data)} 条记录")
    logger.info(f"数据列: {list(data.columns)}")
    
    return data


def preprocess_data(data, lottery_name, window_size=None):
    """预处理数据
    
    :param window_size: 时间窗口大小，默认从配置读取
    """
    # 如果未指定 window_size，从配置读取
    if window_size is None:
        window_size = model_args[lottery_name]["model_args"]["windows_size"]
    
    if lottery_name == "ssq":
        # 双色球：6个红球 + 1个蓝球
        red_cols = [f"红球_{i+1}" for i in range(6)]
        blue_cols = ["蓝球"]
    else:  # dlt
        # 大乐透：5个红球 + 2个蓝球
        red_cols = [f"红球_{i+1}" for i in range(5)]
        blue_cols = [f"蓝球_{i+1}" for i in range(2)]
    
    red_data = data[red_cols].values
    blue_data = data[blue_cols].values
    
    # 创建序列数据
    x_red, y_red = [], []
    x_blue, y_blue = [], []
    
    for i in range(len(data) - window_size):
        # 红球
        x_red.append(red_data[i:i+window_size].flatten())
        y_red.append(red_data[i+window_size])
        
        # 蓝球
        x_blue.append(blue_data[i:i+window_size].flatten())
        y_blue.append(blue_data[i+window_size])
    
    x_red = np.array(x_red)
    y_red = np.array(y_red)
    x_blue = np.array(x_blue)
    y_blue = np.array(y_blue)
    
    logger.info(f"红球数据形状: X={x_red.shape}, y={y_red.shape}")
    logger.info(f"蓝球数据形状: X={x_blue.shape}, y={y_blue.shape}")
    
    return x_red, y_red, x_blue, y_blue


def build_frequency_features(red_data, window_size, n_class):
    """为每个样本构建历史频率特征（在样本目标期之前的全部历史出现频率）
    
    向量化实现：使用前缀累计计数，时间复杂度 O(N * num_balls * n_class) 而非 O(N²)

    red_data: ndarray shape (num_draws, num_balls)
    返回与序列样本对齐的频率矩阵 shape (num_samples, n_class)
    """
    num_draws = red_data.shape[0]
    num_balls = red_data.shape[1]
    samples = num_draws - window_size
    
    if samples <= 0:
        return np.zeros((0, n_class), dtype=np.float32)
    
    # 构建前缀累计计数矩阵 shape: (num_draws + 1, n_class)
    # prefix_counts[i, c] 表示前 i 期（不含第 i 期）中 c+1 号球的出现次数
    prefix_counts = np.zeros((num_draws + 1, n_class), dtype=np.float32)
    
    # one-hot 累加：每期的红球转为 one-hot 并累加
    for draw_idx in range(num_draws):
        # 当期红球的索引（0-indexed）
        ball_indices = red_data[draw_idx] - 1  # shape: (num_balls,)
        one_hot = np.zeros(n_class, dtype=np.float32)
        one_hot[ball_indices] = 1.0
        prefix_counts[draw_idx + 1] = prefix_counts[draw_idx] + one_hot
    
    # 对每个样本，目标期为 i + window_size，取到目标期之前的累计计数
    freq_features = np.zeros((samples, n_class), dtype=np.float32)
    for i in range(samples):
        target_idx = i + window_size
        counts = prefix_counts[target_idx]  # 目标期之前的累计计数
        total = counts.sum()
        if total > 0:
            freq_features[i] = counts / total
        else:
            freq_features[i] = counts
    
    return freq_features


def split_data(x, y, train_ratio=0.6, val_ratio=0.2):
    """划分训练、验证和测试数据
    
    按时间顺序切分，避免数据泄漏。
    
    :param x: 特征数据
    :param y: 标签数据
    :param train_ratio: 训练集占比 (默认 60%)
    :param val_ratio: 验证集占比 (默认 20%)
    :return: x_train, x_val, x_test, y_train, y_val, y_test
    """
    n = len(x)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    x_train = x[:train_idx]
    x_val = x[train_idx:val_idx]
    x_test = x[val_idx:]
    
    y_train = y[:train_idx]
    y_val = y[train_idx:val_idx]
    y_test = y[val_idx:]
    
    logger.info(f"训练数据: {len(x_train)}, 验证数据: {len(x_val)}, 测试数据: {len(x_test)}")
    
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_red_ball_model(lottery_name, x_red, y_red):
    """训练红球模型"""
    logger.info("=" * 50)
    logger.info(f"开始训练【{name_path[lottery_name]['name']}】红球模型...")
    logger.info("=" * 50)
    
    m_args = model_args[lottery_name]
    
    # 划分数据：按时间顺序切分为 train/val/test
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        x_red, y_red, train_ratio=0.6, val_ratio=0.2
    )
    
    # 确定输出类别数
    if lottery_name == "ssq":
        n_class = m_args["model_args"]["red_n_class"]
    else:
        n_class = m_args["model_args"]["red_n_class"]
    
    # 多选集合场景（将红球视为集合）
    # 计算球数量
    num_balls = 6 if lottery_name == "ssq" else 5
    # 计算窗口大小（x_train 列数 = window_size * num_balls）
    window_size = int(x_train.shape[1] / num_balls)

    # 重新 reshape 为 (N, window_size, num_balls)
    x_train_reshaped = x_train.reshape(len(x_train), window_size, num_balls).astype(np.int32)
    x_test_reshaped = x_test.reshape(len(x_test), window_size, num_balls).astype(np.int32)

    # 对输入数据进行减1处理，转换为0-indexed（红球值范围1-33 -> 0-32）
    x_train_reshaped = x_train_reshaped - 1
    x_test_reshaped = x_test_reshaped - 1

    # 准备多标签二值目标（每个样本为长度 n_class 的 0/1 向量）
    y_train_binary = np.zeros((y_train.shape[0], n_class), dtype=np.float32)
    y_test_binary = np.zeros((y_test.shape[0], n_class), dtype=np.float32)
    for i in range(y_train.shape[0]):
        # y_train[i] 包含 num_balls 个数值（1-indexed）
        indices = (y_train[i] - 1).astype(np.int32)
        y_train_binary[i, indices] = 1.0
    for i in range(y_test.shape[0]):
        indices = (y_test[i] - 1).astype(np.int32)
        y_test_binary[i, indices] = 1.0

    # 计算频率特征
    # 需要原始 red_data 来构建频率；重新加载整个数据来获取完整红球历史
    data_full = load_data(lottery_name)
    red_cols = [f"红球_{i+1}" for i in range(num_balls)]
    full_red = data_full[red_cols].values.astype(np.int32)
    freq_features = build_frequency_features(full_red, window_size, n_class)

    # 将频率特征按 train/val/test 划分
    train_end = len(x_train)
    val_end = train_end + len(x_val)
    x_train_freq = freq_features[:train_end]
    x_val_freq = freq_features[train_end:val_end]
    x_test_freq = freq_features[val_end:val_end + len(x_test)]

    # 建立多标签模型
    model = MultiLabelLSTM(
        n_class=n_class,
        window_size=window_size,
        num_balls=num_balls,
        embedding_size=m_args["model_args"]["red_embedding_size"],
        hidden_size=m_args["model_args"]["red_hidden_size"],
        num_layers=m_args["model_args"]["red_layer_size"],
        dropout_rate=0.2,
        use_numeric_features=True,
        num_numeric_features=n_class,
        bidirectional=True
    )
    model.compile_model(learning_rate=m_args["train_args"]["red_learning_rate"])
    model.summary()

    # 确保模型保存目录存在并准备回调
    model_dir = m_args["path"]["red"]
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, model_names["red"])

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_file,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )

    # 训练（多标签接口）
    # 训练时传入两个输入 (序列, 频率特征)
    # 使用真正的验证集而非测试集，避免数据泄漏
    # 准备验证集数据
    x_val_reshaped = x_val.reshape(len(x_val), window_size, num_balls).astype(np.int32) - 1
    y_val_binary = np.zeros((y_val.shape[0], n_class), dtype=np.float32)
    for i in range(y_val.shape[0]):
        indices = (y_val[i] - 1).astype(np.int32)
        y_val_binary[i, indices] = 1.0
    
    model.train(
        [x_train_reshaped, x_train_freq], y_train_binary,
        x_val=[x_val_reshaped, x_val_freq], y_val=y_val_binary,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10,
        callbacks=[checkpoint_cb, reduce_lr_cb]
    )

    # 在真正的测试集上评估（集合命中 metrics）
    x_test_reshaped = x_test.reshape(len(x_test), window_size, num_balls).astype(np.int32) - 1
    proba = model.predict_proba([x_test_reshaped, x_test_freq])  # shape (N, n_class)
    hits = []
    eval_d = {}
    for i in range(len(proba)):
        top_pred = np.argsort(-proba[i])[:num_balls] + 1  # 1-indexed
        true = y_test[i]
        count = len(set(top_pred.tolist()) & set(true.tolist()))
        hits.append(count)
        eval_d[count] = eval_d.get(count, 0) + 1

    avg_hits = np.mean(hits)
    logger.info(f"红球平均命中数 (Top-{num_balls}): {avg_hits:.4f}")
    for k, v in sorted(eval_d.items()):
        logger.info(f"命中{k}个球，{v}期，占比: {round(v * 100 / len(proba), 2)}%")

    # ModelCheckpoint 已经保存了最优模型，若不存在则兜底保存
    if os.path.exists(os.path.join(m_args["path"]["red"], model_names["red"])):
        logger.info(f"红球模型已保存到: {os.path.join(m_args['path']['red'], model_names['red'])}")
    else:
        model.save(os.path.join(m_args["path"]["red"], model_names["red"]))

    return model


def train_blue_ball_model(lottery_name, x_blue, y_blue):
    """训练蓝球模型"""
    logger.info("=" * 50)
    logger.info(f"开始训练【{name_path[lottery_name]['name']}】蓝球模型...")
    logger.info("=" * 50)
    
    m_args = model_args[lottery_name]
    
    # 划分数据：按时间顺序切分为 train/val/test
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        x_blue, y_blue, train_ratio=0.6, val_ratio=0.2
    )
    
    # 确定输出类别数
    if lottery_name == "ssq":
        n_class = m_args["model_args"]["blue_n_class"]
    else:
        n_class = m_args["model_args"]["blue_n_class"]
    
    # 根据玩法选择模型类型：
    # - 双色球(ssq)：单输出分类模型 (LotteryPredictor)
    # - 大乐透(dlt)：两个蓝球 -> 使用 MultiOutputLSTM 多输出模型
    model_dir = m_args["path"]["blue"]
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, model_names["blue"])

    # 确定蓝球输出个数
    if lottery_name == "ssq":
        num_blue_balls = 1
    else:
        num_blue_balls = 2

    if lottery_name == "dlt":
        # 多输出模型
        model = MultiOutputLSTM(
            n_class=n_class,
            window_size=int(x_train.shape[1] / num_blue_balls),
            num_balls=num_blue_balls,
            embedding_size=m_args["model_args"]["blue_embedding_size"],
            hidden_size=m_args["model_args"]["blue_hidden_size"],
            num_layers=m_args["model_args"]["blue_layer_size"],
            dropout_rate=0.2
        )
        model.compile_model(learning_rate=m_args["train_args"]["blue_learning_rate"])
        model.summary()

        # 转换为 0-indexed 并 reshape 为 (N, window_size, num_blue_balls)
        x_train_proc = (x_train - 1).reshape(len(x_train), int(x_train.shape[1] / num_blue_balls), num_blue_balls)
        x_test_proc = (x_test - 1).reshape(len(x_test), int(x_test.shape[1] / num_blue_balls), num_blue_balls)

        # y_train/y_test 为 (N, num_blue_balls)，拆分为列表
        y_train_list = [y_train[:, i] - 1 for i in range(num_blue_balls)]
        y_test_list = [y_test[:, i] - 1 for i in range(num_blue_balls)]

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_file,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )

        # 准备验证集数据
        x_val_proc = (x_val - 1).reshape(len(x_val), int(x_val.shape[1] / num_blue_balls), num_blue_balls)
        y_val_list = [y_val[:, i] - 1 for i in range(num_blue_balls)]

        model.train(
            x_train_proc, y_train_list,
            x_val=x_val_proc,
            y_val_list=y_val_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=10,
            callbacks=[checkpoint_cb, reduce_lr_cb]
        )

        # 在真正的测试集上评估
        x_test_proc = (x_test - 1).reshape(len(x_test), int(x_test.shape[1] / num_blue_balls), num_blue_balls)
        logger.info("蓝球多输出模型训练完成")
        if os.path.exists(model_file):
            logger.info(f"蓝球模型已保存到: {model_file}")
        else:
            model.save(model_file)

        return model

    else:
        # 单输出（ssq）保持原实现
        model = LotteryPredictor(
            n_class=n_class,
            sequence_len=x_train.shape[1],
            embedding_size=m_args["model_args"]["blue_embedding_size"],
            hidden_size=m_args["model_args"]["blue_hidden_size"],
            num_layers=m_args["model_args"]["blue_layer_size"],
            dropout_rate=0.2
        )

        model.compile_model(learning_rate=m_args["train_args"]["blue_learning_rate"])
        model.summary()

        # 对输入数据进行减1处理，转换为0-indexed（蓝球值范围1-16 -> 0-15）
        x_train_proc = x_train - 1
        x_test_proc = x_test - 1

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_file,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )

        # 准备验证集数据
        x_val_proc = x_val - 1

        model.train(
            x_train_proc, y_train - 1,
            x_val=x_val_proc,
            y_val=y_val - 1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=10,
            callbacks=[checkpoint_cb, reduce_lr_cb]
        )

        # 在真正的测试集上评估
        x_test_proc = x_test - 1
        evaluate_model(model, x_test_proc, y_test - 1, name="蓝球")

        if os.path.exists(model_file):
            logger.info(f"蓝球模型已保存到: {model_file}")
        else:
            model.save(model_file)

        return model


def evaluate_model(model, x_test, y_test, name="模型"):
    """评估模型"""
    logger.info(f"\n【{name}】模型评估...")
    
    # 获取预测
    predictions = model.model.predict(x_test, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = np.mean(pred_labels == y_test)
    logger.info(f"【{name}】测试准确率: {accuracy:.4f}")
    
    # 计算命中信息（前N个数字匹配）
    for top_k in [1, 3, 5]:
        top_pred = np.argsort(-predictions, axis=1)[:, :top_k]
        hit_count = np.sum([y_test[i] in top_pred[i] for i in range(len(y_test))])
        hit_rate = hit_count / len(y_test)
        logger.info(f"【{name}】Top-{top_k} 命中率: {hit_rate:.4f}")


def main():
    """主训练函数"""
    logger.info(f"\n开始训练【{name_path[args.name]['name']}】模型")
    logger.info(f"配置: 训练集占比={args.train_test_split}, Batch Size={args.batch_size}, Epochs={args.epochs}")
    
    # 加载数据
    data = load_data(args.name)
    if data is None:
        return
    
    # 预处理数据
    x_red, y_red, x_blue, y_blue = preprocess_data(data, args.name)
    
    # 训练红球模型
    red_model = train_red_ball_model(args.name, x_red, y_red)
    
    # 训练蓝球模型
    blue_model = train_blue_ball_model(args.name, x_blue, y_blue)
    
    logger.info("\n" + "=" * 50)
    logger.info(f"【{name_path[args.name]['name']}】训练完成！")
    logger.info("=" * 50)
    logger.info("下一步: python run_predict.py --name {}".format(args.name))


if __name__ == "__main__":
    if args.name not in ["ssq", "dlt"]:
        raise ValueError("玩法名称无效！请选择 'ssq' (双色球) 或 'dlt' (大乐透)")
    main()
