# -*- coding:utf-8 -*-
"""
快速预测脚本：用于在蓝球模型不可用时，使用已训练的红球多标签模型
并用历史频率启发式选择蓝球（Top-2）。
"""
import os
import numpy as np
import pandas as pd
from loguru import logger

from config import name_path, model_args, model_names, data_file_name
from modeling import MultiLabelLSTM

logger.add("logs/predict_quick_{time}.log", rotation="50 MB", retention="7 days")


def load_data_from_csv(lottery_name):
    data_path = os.path.join(name_path[lottery_name]["path"], data_file_name)
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return None
    df = pd.read_csv(data_path)
    return df


def preprocess_for_predict_quick(df, lottery_name, window_size=3):
    if lottery_name == 'ssq':
        red_cols = [f"红球_{i+1}" for i in range(6)]
        blue_cols = ["蓝球"]
    else:
        red_cols = [f"红球_{i+1}" for i in range(5)]
        blue_cols = [f"蓝球_{i+1}" for i in range(2)]

    if len(df) < window_size:
        window_size = len(df)

    red_data = df[red_cols].values[-window_size:].astype(np.int32) - 1
    blue_data = df[blue_cols].values[-window_size:].astype(np.int32) - 1

    # 频率
    full_red = df[red_cols].values.astype(np.int32)
    hist = full_red.flatten() if full_red.size > 0 else np.array([], dtype=np.int32)
    n_class = model_args[lottery_name]["model_args"]["red_n_class"]
    counts = np.bincount(hist - 1, minlength=n_class).astype(np.float32) if hist.size > 0 else np.zeros(n_class, dtype=np.float32)
    total = counts.sum()
    freq = (counts / total) if total > 0 else counts

    return red_data, blue_data, freq.reshape(1, -1)


def heuristic_blue_from_history(df, lottery_name, k=2):
    if lottery_name == 'ssq':
        blue_cols = ["蓝球"]
    else:
        blue_cols = [f"蓝球_{i+1}" for i in range(2)]
    full_blue = df[blue_cols].values.astype(np.int32)
    hist = full_blue.flatten() if full_blue.size > 0 else np.array([], dtype=np.int32)
    if hist.size == 0:
        return [1] * k
    counts = np.bincount(hist - 1)
    topk = np.argsort(-counts)[:k] + 1
    return sorted(topk.tolist())


def main():
    lottery = 'dlt'
    logger.info('开始快速预测（dlt）')

    df = load_data_from_csv(lottery)
    if df is None:
        return

    # 用配置的 window_size
    win = model_args[lottery]["model_args"].get("windows_size", 3)
    red_data, blue_data, red_freq = preprocess_for_predict_quick(df, lottery, window_size=win)

    # 加载红球模型
    m_args = model_args[lottery]
    red_model_path = os.path.join(m_args["path"]["red"], model_names["red"])
    if not os.path.exists(red_model_path):
        logger.error(f"红球模型不存在: {red_model_path}")
        return

    # 处理路径中可能包含非 ASCII 字符导致的 HDF5/TF 加载问题：
    # 若路径包含非 ASCII 字符，复制模型到临时 ASCII 路径后再加载。
    def _ensure_ascii_path(path):
        try:
            path.encode('ascii')
            return path
        except UnicodeEncodeError:
            import shutil, tempfile
            tmp_dir = os.path.join(tempfile.gettempdir(), 'lottery_models')
            os.makedirs(tmp_dir, exist_ok=True)
            dst = os.path.join(tmp_dir, os.path.basename(path))
            shutil.copy2(path, dst)
            logger.info(f"模型路径包含非 ASCII 字符，已复制到临时路径: {dst}")
            return dst

    red_model = MultiLabelLSTM(
        n_class=m_args["model_args"]["red_n_class"],
        window_size=win,
        num_balls=5,
        embedding_size=m_args["model_args"]["red_embedding_size"],
        hidden_size=m_args["model_args"]["red_hidden_size"],
        num_layers=m_args["model_args"]["red_layer_size"],
        use_numeric_features=True,
        num_numeric_features=m_args["model_args"]["red_n_class"]
    )
    load_path = _ensure_ascii_path(red_model_path)
    # load (modeling.MultiLabelLSTM.load 里有兼容回退)
    red_model.load(load_path)

    # 预测红球 Top-5
    proba = red_model.predict_proba([red_data.reshape(1, red_data.shape[0], red_data.shape[1]), red_freq])
    # proba shape (1, n_class)
    red_preds = np.argsort(-proba[0])[:5] + 1  # 1-indexed
    red_numbers = sorted(red_preds.tolist())

    # 蓝球启发式
    blue_numbers = heuristic_blue_from_history(df, lottery, k=2)

    logger.info(f"预测红球: {red_numbers}")
    logger.info(f"预测蓝球(启发式): {blue_numbers}")
    print("完整预测: 红球 {} 蓝球 {}".format(" ".join(f"{n:02d}" for n in red_numbers), " ".join(f"{n:02d}" for n in blue_numbers)))


if __name__ == '__main__':
    main()
