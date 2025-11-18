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
from modeling import LotteryPredictor, MultiOutputLSTM

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


def preprocess_data(data, lottery_name, window_size=3):
    """预处理数据"""
    
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


def split_data(x, y, split_ratio=0.7):
    """划分训练和测试数据"""
    split_idx = int(len(x) * split_ratio)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"训练数据: {len(x_train)}, 测试数据: {len(x_test)}")
    
    return x_train, x_test, y_train, y_test


def train_red_ball_model(lottery_name, x_red, y_red):
    """训练红球模型"""
    logger.info("=" * 50)
    logger.info(f"开始训练【{name_path[lottery_name]['name']}】红球模型...")
    logger.info("=" * 50)
    
    m_args = model_args[lottery_name]
    
    # 划分数据
    x_train, x_test, y_train, y_test = split_data(
        x_red, y_red, args.train_test_split
    )
    
    # 确定输出类别数
    if lottery_name == "ssq":
        n_class = m_args["model_args"]["red_n_class"]
    else:
        n_class = m_args["model_args"]["red_n_class"]
    
    # 多输出场景（红球通常是多个球）
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

    # 准备多输出标签列表（每个输出为一维向量，0-indexed）
    y_train_list = [ (y_train[:, i] - 1).astype(np.int32) for i in range(num_balls) ]
    y_test_list = [ (y_test[:, i] - 1).astype(np.int32) for i in range(num_balls) ]

    # 建立多输出模型
    model = MultiOutputLSTM(
        n_class=n_class,
        window_size=window_size,
        num_balls=num_balls,
        embedding_size=m_args["model_args"]["red_embedding_size"],
        hidden_size=m_args["model_args"]["red_hidden_size"],
        num_layers=m_args["model_args"]["red_layer_size"],
        dropout_rate=0.2
    )
    model.compile_model(learning_rate=m_args["train_args"]["red_learning_rate"])
    model.summary()

    # 训练（多输出接口）
    model.train(
        x_train_reshaped, y_train_list,
        x_val=x_test_reshaped, y_val_list=y_test_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10
    )

    # 评估（逐个球计算准确率）
    preds = model.predict(x_test_reshaped)
    for i in range(num_balls):
        acc = np.mean(preds[i] == y_test_list[i])
        logger.info(f"红球第{i+1}个位置 测试准确率: {acc:.4f}")

    # 保存模型
    model_dir = m_args["path"]["red"]
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, model_names["red"])
    model.save(model_file)

    return model


def train_blue_ball_model(lottery_name, x_blue, y_blue):
    """训练蓝球模型"""
    logger.info("=" * 50)
    logger.info(f"开始训练【{name_path[lottery_name]['name']}】蓝球模型...")
    logger.info("=" * 50)
    
    m_args = model_args[lottery_name]
    
    # 划分数据
    x_train, x_test, y_train, y_test = split_data(
        x_blue, y_blue, args.train_test_split
    )
    
    # 确定输出类别数
    if lottery_name == "ssq":
        n_class = m_args["model_args"]["blue_n_class"]
    else:
        n_class = m_args["model_args"]["blue_n_class"]
    
    # 建立模型
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
    x_train = x_train - 1
    x_test = x_test - 1
    
    # 训练
    model.train(
        x_train, y_train - 1,  # 转换为 0-indexed
        x_val=x_test,
        y_val=y_test - 1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10
    )
    
    # 评估
    evaluate_model(model, x_test, y_test - 1, name="蓝球")
    
    # 保存模型
    model_dir = m_args["path"]["blue"]
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, model_names["blue"])
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
    if args.train_test_split < 0.5:
        raise ValueError("训练集占比必须 > 0.5!")
    main()



def train_with_eval_blue_ball_model(name, x_train, y_train, x_test, y_test):
    """ 蓝球模型训练与评估 """
    m_args = model_args[name]
    x_train = x_train - 1
    train_data_len = x_train.shape[0]
    if name == "ssq":
        x_train = x_train.reshape(len(x_train), m_args["model_args"]["windows_size"])
        y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=m_args["model_args"]["blue_n_class"])
    else:
        y_train = y_train - 1
    logger.info("训练特征数据维度: {}".format(x_train.shape))
    logger.info("训练标签数据维度: {}".format(y_train.shape))

    x_test = x_test - 1
    test_data_len = x_test.shape[0]
    if name == "ssq":
        x_test = x_test.reshape(len(x_test), m_args["model_args"]["windows_size"])
        y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes=m_args["model_args"]["blue_n_class"])
    else:
        y_test = y_test - 1
    logger.info("训练特征数据维度: {}".format(x_test.shape))
    logger.info("训练标签数据维度: {}".format(y_test.shape))

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        if name == "ssq":
            blue_ball_model = SignalLstmModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                outputs_size=m_args["model_args"]["blue_n_class"],
                layer_size=m_args["model_args"]["blue_layer_size"]
            )
        else:
            blue_ball_model = LstmWithCRFModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                ball_num=m_args["model_args"]["blue_sequence_len"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                words_size=m_args["model_args"]["blue_n_class"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                layer_size=m_args["model_args"]["blue_layer_size"]
            )
        train_step = tf.compat.v1.train.AdamOptimizer(
            learning_rate=m_args["train_args"]["blue_learning_rate"],
            beta1=m_args["train_args"]["blue_beta1"],
            beta2=m_args["train_args"]["blue_beta2"],
            epsilon=m_args["train_args"]["blue_epsilon"],
            use_locking=False,
            name='Adam'
        ).minimize(blue_ball_model.loss)
        sess.run(tf.compat.v1.global_variables_initializer())
        sequence_len = "" if name == "ssq" else m_args["model_args"]["blue_sequence_len"]
        for epoch in range(m_args["model_args"]["blue_epochs"]):
            for i in range(train_data_len):
                if name == "ssq":
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_label
                    ], feed_dict={
                        "inputs:0": x_train[i:(i+1), :],
                        "tag_indices:0": y_train[i:(i+1), :],
                    })
                    if i % 100 == 0:
                        logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, np.argmax(y_train[i:(i+1), :][0]) + 1, pred[0] + 1)
                        )
                else:
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_sequence
                    ], feed_dict={
                        "inputs:0": x_train[i:(i + 1), :, :],
                        "tag_indices:0": y_train[i:(i + 1), :],
                        "sequence_length:0": np.array([sequence_len] * 1)
                    })
                    if i % 100 == 0:
                        logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, y_train[i:(i + 1), :][0] + 1, pred[0] + 1)
                        )
        logger.info("训练耗时: {}".format(time.time() - start_time))
        pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
        if not os.path.exists(m_args["path"]["blue"]):
            os.mkdir(m_args["path"]["blue"])
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["blue"], blue_ball_model_name, extension))
        logger.info("模型评估【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        all_true_count = 0
        for j in range(test_data_len):
            if name == "ssq":
                true = y_test[j:(j + 1), :]
                pred = sess.run(blue_ball_model.pred_label
                , feed_dict={"inputs:0": x_test[j:(j + 1), :]})
            else:
                true = y_test[j:(j + 1), :]
                pred = sess.run(blue_ball_model.pred_sequence
                , feed_dict={
                    "inputs:0": x_test[j:(j + 1), :, :],
                    "sequence_length:0": np.array([sequence_len] * 1)
                })
            count = np.sum(true == pred + 1)
            all_true_count += count
            if count in eval_d:
                eval_d[count] += 1
            else:
                eval_d[count] = 1
        logger.info("测试期数: {}".format(test_data_len))
        for k, v in eval_d.items():
            logger.info("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))
        if name == "ssq":
            logger.info(
                "整体准确率: {}%".format(
                    round(all_true_count * 100 / test_data_len, 2)
                )
            )
        else:
            logger.info(
                "整体准确率: {}%".format(
                    round(all_true_count * 100 / (test_data_len * sequence_len), 2)
                )
            )


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        main()
