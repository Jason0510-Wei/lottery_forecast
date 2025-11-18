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
from get_data import get_current_number, spider
from modeling import LotteryPredictor, MultiOutputLSTM

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
    
    red_model = MultiOutputLSTM(
        n_class=m_args["model_args"]["red_n_class"],
        window_size=m_args["model_args"]["windows_size"],
        num_balls=num_red_balls,
        embedding_size=m_args["model_args"]["red_embedding_size"],
        hidden_size=m_args["model_args"]["red_hidden_size"],
        num_layers=m_args["model_args"]["red_layer_size"]
    )
    red_model.load(red_model_path)
    logger.info("✓ 红球模型已加载")
    
    # 加载蓝球模型（单输出）
    blue_model_path = os.path.join(m_args["path"]["blue"], model_names["blue"])
    if not os.path.exists(blue_model_path):
        logger.error(f"蓝球模型不存在: {blue_model_path}")
        return None, None
    
    if lottery_name == "ssq":
        num_blue_balls = 1
    else:
        num_blue_balls = 2
    
    blue_model = MultiOutputLSTM(
        n_class=m_args["model_args"]["blue_n_class"],
        window_size=m_args["model_args"]["windows_size"],
        num_balls=num_blue_balls,
        embedding_size=m_args["model_args"]["blue_embedding_size"],
        hidden_size=m_args["model_args"]["blue_hidden_size"],
        num_layers=m_args["model_args"]["blue_layer_size"]
    )
    blue_model.load(blue_model_path)
    logger.info("✓ 蓝球模型已加载")
    
    return red_model, blue_model


def get_predict_data(lottery_name):
    """获取最新数据进行预测"""
    logger.info(f"获取【{name_path[lottery_name]['name']}】最新数据...")
    
    # 获取最新期号
    current_number = get_current_number(lottery_name)
    if current_number is None:
        logger.error("无法获取最新期号")
        return None
    
    logger.info(f"最新期号: {current_number}")
    
    # 爬取从今年第一期（25001）到最新期的所有数据
    try:
        data = spider(lottery_name, 25001, current_number, "predict")
        if data is None or len(data) == 0:
            logger.error("未能获取预测数据")
            return None
        
        logger.info(f"获取了 {len(data)} 条最近数据（从25001期到{current_number}期）")
        return data
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return None


def preprocess_predict_data(data, lottery_name, window_size=3):
    """预处理预测数据"""
    
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
    
    return red_data, blue_data


def predict_next_draw(red_model, blue_model, red_data, blue_data, lottery_name):
    """预测下一期开奖"""
    logger.info("\n" + "=" * 60)
    logger.info(f"【{name_path[lottery_name]['name']}】预测结果")
    logger.info("=" * 60)
    
    # 预测红球（多输出模型）
    logger.info("预测红球...")
    red_preds = red_model.predict(red_data.reshape(1, red_data.shape[0], red_data.shape[1]))
    
    # 预测蓝球（多输出模型）
    logger.info("预测蓝球...")
    blue_preds = blue_model.predict(blue_data.reshape(1, blue_data.shape[0], blue_data.shape[1]))
    
    # 组合结果
    if lottery_name == "ssq":
        # 双色球：6个红球 + 1个蓝球
        red_numbers = sorted([red_preds[i][0] for i in range(6)])
        # 蓝球模型是单输出，直接取值
        if isinstance(blue_preds, list):
            blue_number = blue_preds[0][0]  # 多输出模型
        else:
            blue_number = blue_preds[0]    # 单输出模型
        
        logger.info(f"\n【预测红球】: {red_numbers}")
        logger.info(f"【预测蓝球】: {blue_number}")
        logger.info("\n【完整预测】: 红球 {}, 蓝球 {}".format(
            " ".join(f"{n:02d}" for n in red_numbers), f"{blue_number:02d}"
        ))
        
        return red_numbers, blue_number
    
    else:  # dlt
        # 大乐透：5个红球 + 2个蓝球
        red_numbers = sorted([red_preds[i][0] for i in range(5)])
        # 蓝球模型可能是单输出或多输出
        if isinstance(blue_preds, list):
            blue_numbers = sorted([blue_preds[i][0] for i in range(2)])
        else:
            blue_numbers = sorted([blue_preds[i] for i in range(2)])
        
        logger.info(f"\n【预测红球】: {red_numbers}")
        logger.info(f"【预测蓝球】: {blue_numbers}")
        logger.info("\n【完整预测】: 红球 {}, 蓝球 {}".format(
            " ".join(f"{n:02d}" for n in red_numbers),
            " ".join(f"{n:02d}" for n in blue_numbers)
        ))
        
        return red_numbers, blue_numbers


def get_top_predictions(red_model, blue_model, red_data, blue_data, lottery_name, top_k=5):
    """获取预测的Top K个可能"""
    logger.info(f"\n【Top {top_k} 可能性】")
    logger.info("-" * 60)
    
    # 使用正确的输入形状获取概率
    red_proba = red_model.predict_proba(red_data.reshape(1, red_data.shape[0], red_data.shape[1]))
    blue_proba = blue_model.predict_proba(blue_data.reshape(1, blue_data.shape[0], blue_data.shape[1]))
    
    # 获取 Top K
    if isinstance(red_proba, list):
        # 多输出模型：取第一个输出的概率
        red_top_k = np.argsort(-red_proba[0][0])[:top_k] + 1
    else:
        # 单输出模型
        red_top_k = np.argsort(-red_proba[0])[:top_k] + 1
    
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
    red_data, blue_data = preprocess_predict_data(data, args.name)
    
    # 进行预测
    red_pred, blue_pred = predict_next_draw(red_model, blue_model, red_data, blue_data, args.name)
    
    # 获取Top K预测
    get_top_predictions(red_model, blue_model, red_data, blue_data, args.name, top_k=5)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("\n⚠️  免责声明: 此预测仅供娱乐参考，不构成投资建议")
    logger.info("彩票具有风险性，请理性购彩\n")


if __name__ == "__main__":
    if args.name not in ["ssq", "dlt"]:
        raise ValueError("玩法名称无效！请选择 'ssq' (双色球) 或 'dlt' (大乐透)")
    main()
