# -*- coding: utf-8 -*-
"""
彩票预测系统配置文件
Author: Updated Version
"""
import os

# 彩票类型配置
ball_name = [
    ("红球", "red"),
    ("蓝球", "blue")
]

data_file_name = "data.csv"

# 数据路径配置
name_path = {
    "ssq": {
        "name": "双色球",
        "path": "data/ssq/"
    },
    "dlt": {
        "name": "大乐透",
        "path": "data/dlt/"
    }
}

model_path = os.path.join(os.getcwd(), "model")

# 模型参数配置
model_args = {
    "ssq": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 32,
            "sequence_len": 6,
            "red_n_class": 33,
            "red_epochs": 50,
            "red_embedding_size": 64,
            "red_hidden_size": 128,
            "red_layer_size": 2,
            "blue_n_class": 16,
            "blue_epochs": 50,
            "blue_embedding_size": 64,
            "blue_hidden_size": 128,
            "blue_layer_size": 2
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "blue_learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 10
        },
        "path": {
            "red": os.path.join(model_path, "ssq/red_ball_model/"),
            "blue": os.path.join(model_path, "ssq/blue_ball_model/")
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 32,
            "red_sequence_len": 5,
            "red_n_class": 35,
            "red_epochs": 50,
            "red_embedding_size": 64,
            "red_hidden_size": 128,
            "red_layer_size": 2,
            "blue_sequence_len": 2,
            "blue_n_class": 12,
            "blue_epochs": 50,
            "blue_embedding_size": 64,
            "blue_hidden_size": 128,
            "blue_layer_size": 2
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "blue_learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 10
        },
        "path": {
            "red": os.path.join(model_path, "dlt/red_ball_model/"),
            "blue": os.path.join(model_path, "dlt/blue_ball_model/")
        }
    }
}

# 模型保存配置
model_names = {
    "red": "red_ball_model.h5",
    "blue": "blue_ball_model.h5"
}

