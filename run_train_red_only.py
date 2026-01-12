# 临时脚本：仅训练红球模型（用于快速验证改动）
from loguru import logger
from run_train_model import load_data, preprocess_data, train_red_ball_model

logger.add("logs/run_train_red_only_{time}.log", rotation="10 MB", retention="7 days")

if __name__ == '__main__':
    name = 'ssq'
    data = load_data(name)
    x_red, y_red, x_blue, y_blue = preprocess_data(data, name)
    # 为了快速验证，只使用前 500 个样本并短跑
    max_samples = 500
    x_red = x_red[:max_samples]
    y_red = y_red[:max_samples]
    # 临时覆盖全局 args 以缩短训练轮数
    import run_train_model as rtm
    rtm.args.epochs = 2
    model = train_red_ball_model(name, x_red, y_red)
    logger.info('短跑红球训练（子集）完成')
