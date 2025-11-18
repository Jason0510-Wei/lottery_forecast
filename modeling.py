# -*- coding:utf-8 -*-
"""
模型定义模块 - 使用现代 TensorFlow 2.x Keras API
Updated Version
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from loguru import logger


class LotteryPredictor:
    """彩票预测模型 - 基于 LSTM 的序列预测"""
    
    def __init__(self, n_class, sequence_len, embedding_size=64, hidden_size=128, 
                 num_layers=2, dropout_rate=0.2):
        """
        初始化预测器
        :param n_class: 输出类别数
        :param sequence_len: 序列长度
        :param embedding_size: 嵌入维度
        :param hidden_size: LSTM 隐层大小
        :param num_layers: LSTM 层数
        :param dropout_rate: Dropout 比例
        """
        self.n_class = n_class
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """构建 LSTM 模型"""
        logger.info("构建模型...")
        
        inputs = keras.Input(shape=(self.sequence_len,), dtype=tf.int32, name="inputs")
        
        # 嵌入层
        x = layers.Embedding(
            input_dim=self.n_class,
            output_dim=self.embedding_size,
            name="embedding"
        )(inputs)
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 多层 LSTM
        for i in range(self.num_layers):
            x = layers.LSTM(
                units=self.hidden_size,
                return_sequences=(i < self.num_layers - 1),
                dropout=self.dropout_rate,
                name=f"lstm_{i}"
            )(x)
        
        # 输出层
        outputs = layers.Dense(
            units=self.n_class,
            activation='softmax',
            name="predictions"
        )(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="LotteryPredictor")
        logger.info("模型构建完成")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("模型编译完成")
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=32,
              early_stopping_patience=10):
        """
        训练模型
        :param x_train: 训练特征
        :param y_train: 训练标签
        :param x_val: 验证特征
        :param y_val: 验证标签
        :param epochs: 训练轮数
        :param batch_size: 批大小
        :param early_stopping_patience: 早停耐心
        """
        if self.model is None:
            self.compile_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        validation_data = (x_val, y_val) if x_val is not None else None
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("模型训练完成")
        return self.history
    
    def predict(self, x):
        """
        预测
        :param x: 输入特征
        :return: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未初始化！")
        
        predictions = self.model.predict(x, verbose=0)
        return np.argmax(predictions, axis=1) + 1  # 返回 1-indexed 的预测值
    
    def predict_proba(self, x):
        """获取预测概率"""
        if self.model is None:
            raise ValueError("模型未初始化！")
        return self.model.predict(x, verbose=0)
    
    def save(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化！")
        self.model.save(filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        self.model = keras.models.load_model(filepath, safe_mode=False)
        logger.info(f"模型已加载: {filepath}")
    
    def summary(self):
        """打印模型摘要"""
        if self.model is None:
            logger.warning("模型未初始化")
            return
        self.model.summary()


class MultiOutputLSTM:
    """多输出 LSTM 模型 - 用于预测多个数字"""
    
    def __init__(self, n_class, window_size, num_balls, embedding_size=64, 
                 hidden_size=128, num_layers=2, dropout_rate=0.2):
        """
        初始化多输出模型
        :param n_class: 每个输出的类别数
        :param window_size: 时间窗口大小
        :param num_balls: 输出球数
        :param embedding_size: 嵌入维度
        :param hidden_size: LSTM 隐层大小
        :param num_layers: LSTM 层数
        :param dropout_rate: Dropout 比例
        """
        self.n_class = n_class
        self.window_size = window_size
        self.num_balls = num_balls
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
    
    def build_model(self):
        """构建多输出 LSTM 模型"""
        logger.info("构建多输出模型...")
        
        inputs = keras.Input(
            shape=(self.window_size, self.num_balls),
            dtype=tf.int32,
            name="inputs"
        )
        
        # 处理每个球的嵌入
        embeddings = []
        for i in range(self.num_balls):
            ball_input = layers.Lambda(lambda x: x[:, :, i])(inputs)
            emb = layers.Embedding(
                input_dim=self.n_class,
                output_dim=self.embedding_size,
                name=f"embedding_{i}"
            )(ball_input)
            embeddings.append(emb)
        
        # 合并嵌入
        if len(embeddings) > 1:
            x = layers.Concatenate()(embeddings)
        else:
            x = embeddings[0]
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 多层 LSTM
        for i in range(self.num_layers):
            x = layers.LSTM(
                units=self.hidden_size,
                return_sequences=(i < self.num_layers - 1),
                dropout=self.dropout_rate,
                name=f"lstm_{i}"
            )(x)
        
        # 多个输出头，每个球一个
        outputs = []
        for i in range(self.num_balls):
            output = layers.Dense(
                units=self.n_class,
                activation='softmax',
                name=f"ball_{i}_output"
            )(x)
            outputs.append(output)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="MultiOutputLSTM")
        logger.info("多输出模型构建完成")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        losses = ['sparse_categorical_crossentropy'] * self.num_balls
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=['accuracy'] * self.num_balls
        )
        logger.info("多输出模型编译完成")
    
    def train(self, x_train, y_train_list, x_val=None, y_val_list=None, 
              epochs=50, batch_size=32, early_stopping_patience=10):
        """训练模型"""
        if self.model is None:
            self.compile_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        validation_data = (x_val, y_val_list) if x_val is not None else None
        
        self.history = self.model.fit(
            x_train, y_train_list,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("多输出模型训练完成")
        return self.history
    
    def predict(self, x):
        """预测"""
        if self.model is None:
            raise ValueError("模型未初始化！")
        
        predictions = self.model.predict(x, verbose=0)
        # 转换为 1-indexed
        if isinstance(predictions, list):
            # 多输出情况
            return [np.argmax(pred, axis=1) + 1 for pred in predictions]
        else:
            # 单输出情况
            return np.argmax(predictions, axis=1) + 1
    
    def predict_proba(self, x):
        """获取预测概率"""
        if self.model is None:
            raise ValueError("模型未初始化！")
        
        predictions = self.model.predict(x, verbose=0)
        return predictions
    
    def save(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化！")
        self.model.save(filepath)
        logger.info(f"多输出模型已保存到: {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        self.model = keras.models.load_model(filepath, safe_mode=False)
        logger.info(f"多输出模型已加载: {filepath}")

    def summary(self):
        """打印模型摘要"""
        if self.model is None:
            logger.warning("模型未初始化")
            return
        self.model.summary()
