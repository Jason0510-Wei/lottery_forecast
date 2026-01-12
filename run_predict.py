# -*- coding:utf-8 -*-
"""
预测脚本 - 使用训练好的模型进行彩票号码预测
Updated Version
"""
import argparse
import os
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

from config import name_path, model_args, model_names, data_file_name
from tensorflow import keras
from get_data import get_current_number, spider
from modeling import LotteryPredictor, MultiOutputLSTM, MultiLabelLSTM
import os.path

# 配置日志
logger.add("logs/predict_{time}.log", rotation="500 MB", retention="7 days")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择预测数据: 双色球(ssq)/大乐透(dlt)")
args = parser.parse_args()


def load_models(lottery_name):
    """加载训练好的模型"""
    logger.info(f"加载【{name_path[lottery_name]['name']}】模型...")
    
    m_args = model_args[lottery_name]
    
    # 加载红球模型（多输出）
    red_model_path = os.path.join(m_args["path"]["red"], model_names["red"])
    if not os.path.exists(red_model_path):
        logger.error(f"红球模型不存在: {red_model_path}")
        logger.info(f"请先运行: python run_train_model.py --name {lottery_name}")
        return None, None
    
    if lottery_name == "ssq":
        num_red_balls = 6
    else:
        num_red_balls = 5
    
    # 加载为多标签模型（如果之前训练采用多标签方式）
    red_model = MultiLabelLSTM(
        n_class=m_args["model_args"]["red_n_class"],
        window_size=m_args["model_args"]["windows_size"],
        num_balls=num_red_balls,
        embedding_size=m_args["model_args"]["red_embedding_size"],
        hidden_size=m_args["model_args"]["red_hidden_size"],
        num_layers=m_args["model_args"]["red_layer_size"]
    )
    try:
        red_model.load(red_model_path)
    except ValueError as e:
        # 旧模型可能包含匿名 Lambda，尝试启用不安全反序列化后重试
        msg = str(e)
        if 'Lambda' in msg or 'deserialization of a `Lambda`' in msg:
            logger.warning("检测到 Lambda 层反序列化限制，启用不安全反序列化并重试加载模型（本地文件可信时可用）")
            keras.config.enable_unsafe_deserialization()
            red_model.load(red_model_path)
        else:
            raise
    logger.info("✓ 红球模型已加载")
    
    # 加载蓝球模型
    blue_model_path = os.path.join(m_args["path"]["blue"], model_names["blue"])
    if not os.path.exists(blue_model_path):
        logger.error(f"蓝球模型不存在: {blue_model_path}")
        return None, None
    
    if lottery_name == "ssq":
        # 双色球蓝球：单输出分类模型，使用 LotteryPredictor
        num_blue_balls = 1
        window_size = m_args["model_args"]["windows_size"]
        blue_model = LotteryPredictor(
            n_class=m_args["model_args"]["blue_n_class"],
            sequence_len=window_size * num_blue_balls,
            embedding_size=m_args["model_args"]["blue_embedding_size"],
            hidden_size=m_args["model_args"]["blue_hidden_size"],
            num_layers=m_args["model_args"]["blue_layer_size"]
        )
    else:
        # 大乐透蓝球：双输出模型，使用 MultiOutputLSTM
        num_blue_balls = 2
        blue_model = MultiOutputLSTM(
            n_class=m_args["model_args"]["blue_n_class"],
            window_size=m_args["model_args"]["windows_size"],
            num_balls=num_blue_balls,
            embedding_size=m_args["model_args"]["blue_embedding_size"],
            hidden_size=m_args["model_args"]["blue_hidden_size"],
            num_layers=m_args["model_args"]["blue_layer_size"]
        )
    try:
        blue_model.load(blue_model_path)
    except ValueError as e:
        msg = str(e)
        if 'Lambda' in msg or 'deserialization of a `Lambda`' in msg:
            logger.warning("检测到 Lambda 层反序列化限制（蓝球模型），启用不安全反序列化并重试加载模型")
            keras.config.enable_unsafe_deserialization()
            blue_model.load(blue_model_path)
        else:
            raise
    logger.info("✓ 蓝球模型已加载")
    
    return red_model, blue_model


def get_predict_data(lottery_name):
    """获取最新数据进行预测
    
    优先使用本地已有数据，如果需要最新数据则增量爬取。
    """
    logger.info(f"获取【{name_path[lottery_name]['name']}】最新数据...")
    
    # 优先读取本地数据文件
    local_data_path = os.path.join(name_path[lottery_name]["path"], data_file_name)
    local_data = None
    if os.path.exists(local_data_path):
        try:
            local_data = pd.read_csv(local_data_path)
            logger.info(f"已加载本地数据: {len(local_data)} 条记录")
        except Exception as e:
            logger.warning(f"读取本地数据失败: {e}")
    
    # 获取最新期号
    current_number = get_current_number(lottery_name)
    if current_number is None:
        if local_data is not None and len(local_data) > 0:
            logger.warning("无法获取最新期号，使用本地缓存数据进行预测")
            return local_data
        logger.error("无法获取最新期号且无本地数据")
        return None
    
    logger.info(f"最新期号: {current_number}")
    
    # 计算起始期号：根据最新期号自动推算（而非硬编码）
    # 期号格式通常为 YYXXX 或 YYYYXXX
    current_str = str(current_number)
    if len(current_str) >= 5:
        # 提取年份前缀和期次后缀
        year_prefix = current_str[:-3]  # 例如 "25" 或 "2025"
        # 起始期号为当年第一期
        start_number = int(year_prefix + "001")
    else:
        # 回退到当前期号减去一定数量
        start_number = max(1, current_number - 150)
    
    # 如果本地数据足够新，直接使用
    if local_data is not None and len(local_data) > 0:
        # 检查本地数据是否包含最新期号
        if "期号" in local_data.columns:
            latest_local = local_data["期号"].max()
            if latest_local >= current_number:
                logger.info("本地数据已是最新，无需爬取")
                return local_data
            else:
                # 只爬取增量数据
                start_number = int(latest_local) + 1
                logger.info(f"将增量爬取从 {start_number} 到 {current_number} 的数据")
    
    # 爬取数据
    try:
        data = spider(lottery_name, start_number, current_number, "predict")
        if data is None or len(data) == 0:
            if local_data is not None and len(local_data) > 0:
                logger.warning("爬取新数据失败，使用本地缓存数据")
                return local_data
            logger.error("未能获取预测数据")
            return None
        
        # 如果有本地数据，合并增量数据
        if local_data is not None and len(local_data) > 0:
            # 合并并去重
            combined = pd.concat([local_data, data], ignore_index=True)
            if "期号" in combined.columns:
                combined = combined.drop_duplicates(subset=["期号"], keep="last")
                combined = combined.sort_values("期号").reset_index(drop=True)
            data = combined
        
        logger.info(f"获取了 {len(data)} 条数据")
        return data
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        if local_data is not None and len(local_data) > 0:
            logger.warning("使用本地缓存数据")
            return local_data
        return None


def preprocess_predict_data(data, lottery_name, window_size=None):
    """预处理预测数据
    
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
    
    # 取最后 window_size 条记录
    if len(data) < window_size:
        logger.warning(f"数据不足 {window_size} 条，使用现有数据")
        window_size = len(data)
    
    red_data = data[red_cols].values[-window_size:].astype(np.int32) - 1
    blue_data = data[blue_cols].values[-window_size:].astype(np.int32) - 1

    # 构建频率特征：统计历史（到当前为止）
    full_red = data[red_cols].values.astype(np.int32)
    hist = full_red.flatten() if full_red.size > 0 else np.array([], dtype=np.int32)
    n_class = model_args[lottery_name]["model_args"]["red_n_class"]
    counts = np.bincount(hist - 1, minlength=n_class).astype(np.float32) if hist.size > 0 else np.zeros(n_class, dtype=np.float32)
    total = counts.sum()
    freq = (counts / total) if total > 0 else counts

    return red_data, blue_data, freq.reshape(1, -1)


def predict_next_draw(red_model, blue_model, red_data, blue_data, lottery_name, red_freq=None):
    """预测下一期开奖"""
    logger.info("\n" + "=" * 60)
    logger.info(f"【{name_path[lottery_name]['name']}】预测结果")
    logger.info("=" * 60)
    
    # 根据玩法确定红球数量
    num_red_balls = 6 if lottery_name == "ssq" else 5
    
    # 预测红球（多标签模型）
    logger.info("预测红球...")
    # red_model 为多标签模型，获取概率并选 Top-K
    if red_freq is not None:
        red_proba = red_model.predict_proba([red_data.reshape(1, red_data.shape[0], red_data.shape[1]), red_freq])
    else:
        red_proba = red_model.predict_proba(red_data.reshape(1, red_data.shape[0], red_data.shape[1]))
    # red_proba shape: (1, n_class)
    # 取 Top-K 并转为 1-indexed，直接得到整数值
    red_preds = np.argsort(-red_proba[0])[:num_red_balls] + 1
    
    # 预测蓝球
    logger.info("预测蓝球...")
    
    # 组合结果
    if lottery_name == "ssq":
        # 双色球：6个红球 + 1个蓝球
        red_numbers = sorted([int(x) for x in red_preds])
        
        # 双色球蓝球模型是 LotteryPredictor（单输出分类），输入为 2D 序列
        blue_input = blue_data.flatten().reshape(1, -1) - 1  # 转为 0-indexed, shape (1, window_size)
        blue_pred = blue_model.predict(blue_input)
        blue_number = int(blue_pred[0])  # predict 返回 1-indexed
        
        logger.info(f"\n【预测红球】: {red_numbers}")
        logger.info(f"【预测蓝球】: {blue_number}")
        logger.info("\n【完整预测】: 红球 {}, 蓝球 {}".format(
            " ".join(f"{n:02d}" for n in red_numbers), f"{blue_number:02d}"
        ))
        
        return red_numbers, blue_number
    
    else:  # dlt
        # 大乐透：5个红球 + 2个蓝球
        red_numbers = sorted([int(x) for x in red_preds])
        
        # 大乐透蓝球模型是 MultiOutputLSTM（双输出），输入为 3D 序列
        blue_input = blue_data.reshape(1, blue_data.shape[0], blue_data.shape[1]) - 1  # 0-indexed
        blue_preds = blue_model.predict(blue_input)
        # blue_preds 是列表，每个元素对应一个输出
        if isinstance(blue_preds, list):
            blue_numbers = sorted([int(blue_preds[i][0]) for i in range(2)])
        else:
            blue_numbers = sorted([int(blue_preds[i]) for i in range(2)])
        
        logger.info(f"\n【预测红球】: {red_numbers}")
        logger.info(f"【预测蓝球】: {blue_numbers}")
        logger.info("\n【完整预测】: 红球 {}, 蓝球 {}".format(
            " ".join(f"{n:02d}" for n in red_numbers),
            " ".join(f"{n:02d}" for n in blue_numbers)
        ))
        
        return red_numbers, blue_numbers


def get_top_predictions(red_model, blue_model, red_data, blue_data, lottery_name, top_k=5, red_freq=None):
    """获取预测的Top K个可能"""
    logger.info(f"\n【Top {top_k} 可能性】")
    logger.info("-" * 60)
    
    # 使用正确的输入形状获取概率
    if red_freq is not None:
        red_proba = red_model.predict_proba([red_data.reshape(1, red_data.shape[0], red_data.shape[1]), red_freq])
    else:
        red_proba = red_model.predict_proba(red_data.reshape(1, red_data.shape[0], red_data.shape[1]))
    
    # 红球概率处理
    if isinstance(red_proba, list):
        # 多输出模型：取第一个输出的概率
        red_top_k = np.argsort(-red_proba[0][0])[:top_k] + 1
    else:
        # 单输出/多标签模型
        red_top_k = np.argsort(-red_proba[0])[:top_k] + 1
    
    # 蓝球概率处理：根据玩法使用不同的输入形状
    if lottery_name == "ssq":
        # 双色球蓝球：LotteryPredictor，输入为 2D
        blue_input = blue_data.flatten().reshape(1, -1) - 1
        blue_proba = blue_model.predict_proba(blue_input)
    else:
        # 大乐透蓝球：MultiOutputLSTM，输入为 3D
        blue_input = blue_data.reshape(1, blue_data.shape[0], blue_data.shape[1]) - 1
        blue_proba = blue_model.predict_proba(blue_input)
    
    if isinstance(blue_proba, list):
        # 多输出模型：取第一个输出的概率
        blue_top_k = np.argsort(-blue_proba[0][0])[:top_k] + 1
    else:
        # 单输出模型
        blue_top_k = np.argsort(-blue_proba[0])[:top_k] + 1
    
    logger.info(f"红球 Top {top_k}: {red_top_k}")
    logger.info(f"蓝球 Top {top_k}: {blue_top_k}")


def main():
    """主预测函数"""
    logger.info(f"\n开始预测【{name_path[args.name]['name']}】\n")
    
    # 加载模型
    red_model, blue_model = load_models(args.name)
    if red_model is None or blue_model is None:
        logger.error("模型加载失败")
        return
    
    # 获取预测数据
    data = get_predict_data(args.name)
    if data is None:
        logger.error("数据获取失败")
        return
    
    # 预处理数据
    red_data, blue_data, red_freq = preprocess_predict_data(data, args.name)

    # 进行预测
    # red_model 现在期望两个输入: 序列和频率特征（如果模型构建时使用了数值特征）
    red_pred, blue_pred = predict_next_draw(red_model, blue_model, red_data, blue_data, args.name, red_freq)
    
    # 获取Top K预测
    get_top_predictions(red_model, blue_model, red_data, blue_data, args.name, top_k=5, red_freq=red_freq)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("\n⚠️  免责声明: 此预测仅供娱乐参考，不构成投资建议")
    logger.info("彩票具有风险性，请理性购彩\n")


if __name__ == "__main__":
    if args.name not in ["ssq", "dlt"]:
        raise ValueError("玩法名称无效！请选择 'ssq' (双色球) 或 'dlt' (大乐透)")
    main()
