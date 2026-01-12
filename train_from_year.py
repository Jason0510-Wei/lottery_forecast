# -*- coding:utf-8 -*-
"""
从指定年份开始训练脚本（不会覆盖原始 data.csv）
- 读取 data/<lottery>/data.csv
- 以期号前两位推断年份（如 '25140' -> 2025），筛选 >= start_year 的记录
- 在内存中把筛选后的 DataFrame 传入 `run_train_model` 模块的训练函数（不改变磁盘上的原 data.csv）
"""
import argparse
import os
import time
import pandas as pd
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='dlt', choices=['ssq', 'dlt'])
parser.add_argument('--start_year', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--train_test_split', type=float, default=0.7)
args = parser.parse_args()

logger.add('logs/train_from_year_{time}.log', rotation='100 MB', retention='7 days')


def infer_year_from_issue(issue_val):
    s = str(issue_val)
    # 保证至少 2 位，部分期号可能已被读为 int 丢失前导零
    if len(s) < 2:
        s = s.zfill(2)
    # 使用前两位作为年份后两位
    prefix = s[:2]
    try:
        yy = int(prefix)
    except Exception:
        # 回退：取全部数字，最后两位
        try:
            yy = int(s[-2:])
        except Exception:
            return None
    return 2000 + yy


def main():
    name = args.name
    start_year = args.start_year
    epochs = args.epochs
    batch_size = args.batch_size
    split = args.train_test_split

    from config import name_path, data_file_name

    data_path = os.path.join(name_path[name]['path'], data_file_name)
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return

    logger.info(f"读取数据: {data_path}")
    df = pd.read_csv(data_path, dtype={'期数': str})

    # 补齐期号字符串并推断年份
    df['期数_str'] = df['期数'].astype(str).str.zfill(5)
    df['year_infer'] = df['期数_str'].str[:2].astype(int) + 2000

    df_filtered = df[df['year_infer'] >= start_year].copy()
    if df_filtered.shape[0] == 0:
        logger.error(f"未找到 >= {start_year} 的记录，数据量为 0，停止。")
        return

    logger.info(f"原始记录数: {len(df)}, 筛选后记录数: {len(df_filtered)} (从 {start_year} 年开始)")

    # 不覆盖原始文件：我们在内存中使用 df_filtered。为让 run_train_model 使用此数据，导入并 monkey-patch
    # 导入 run_train_model 前暂存并清空 sys.argv，防止其 argparse 解析我们额外的参数
    import sys
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    import run_train_model as rtm
    import types
    # 恢复 argv
    sys.argv = saved_argv

    # 覆盖 rtm.args
    rtm.args = argparse.Namespace(
        name=name,
        train_test_split=split,
        batch_size=batch_size,
        epochs=epochs
    )

    # 将 rtm.load_data 替换为返回 df_filtered 的函数
    def _load_data_override(lottery_name):
        logger.info(f"(override) 返回内存中筛选后的数据，shape={df_filtered.shape}")
        return df_filtered

    rtm.load_data = _load_data_override

    # 运行预处理与训练（直接调用 run_train_model 内的函数）
    logger.info("开始预处理数据并训练（使用内存筛选数据）...")
    x_red, y_red, x_blue, y_blue = rtm.preprocess_data(df_filtered, name)

    # 启动红球训练
    logger.info("开始训练红球模型...")
    red_model = rtm.train_red_ball_model(name, x_red, y_red)

    # 启动蓝球训练
    logger.info("开始训练蓝球模型...")
    blue_model = rtm.train_blue_ball_model(name, x_blue, y_blue)

    logger.info("训练（从年份筛选）完成")


if __name__ == '__main__':
    main()
