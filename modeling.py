# -*- coding:utf-8 -*-
"""
模型定义模块 - 使用现代 TensorFlow 2.x Keras API
Updated Version
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os
import sys
from loguru import logger


def _get_safe_path(filepath):
    """将路径转换为 TensorFlow 可安全加载的格式（解决 Windows 中文路径问题）
    
    TensorFlow 底层 C++ 使用 UTF-8 编码处理文件路径，
    在 Windows 上如果路径包含中文可能导致 UnicodeDecodeError。
    此函数尝试使用 Windows 短路径名（8.3格式）来绕过此问题。
    """
    if sys.platform != 'win32':
        return filepath
    
    # 确保路径存在
    if not os.path.exists(filepath):
        return filepath
    
    try:
        import ctypes
        from ctypes import wintypes
        
        # 获取短路径名
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        GetShortPathNameW.restype = wintypes.DWORD
        
        # 先获取需要的缓冲区大小
        length = GetShortPathNameW(filepath, None, 0)
        if length == 0:
            return filepath
        
        # 获取短路径
        buffer = ctypes.create_unicode_buffer(length)
        GetShortPathNameW(filepath, buffer, length)
        short_path = buffer.value
        
        if short_path:
            logger.debug(f"路径转换: {filepath} -> {short_path}")
            return short_path
    except Exception as e:
        logger.debug(f"获取短路径失败: {e}")
    
    return filepath


# 兼容性 LSTM 层：忽略旧版本模型中的 time_major 参数
# 通过 monkey patch 方式直接替换原始 LSTM 类的 __init__ 方法
_original_lstm_init = layers.LSTM.__init__

def _patched_lstm_init(self, *args, **kwargs):
    """兼容性 LSTM 初始化，忽略旧版本 TensorFlow 保存模型中可能包含的已废弃参数"""
    # 移除旧版本 TensorFlow 中可能存在的已废弃参数
    deprecated_params = ['time_major', 'implementation']
    for param in deprecated_params:
        kwargs.pop(param, None)
    _original_lstm_init(self, *args, **kwargs)

# 应用 monkey patch
layers.LSTM.__init__ = _patched_lstm_init


# 可序列化的 Slice 层：从输入 (batch, time, num_balls) 中按索引抽取第 i 列
class SliceLayer(layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[:, :, self.index]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"index": self.index})
        return cfg


# 可序列化的 ReduceSum 层：沿着时间维度求和（用于注意力上下文聚合）
class ReduceSumLayer(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


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
              early_stopping_patience=10, callbacks=None):
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
        
        cb_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        # 将外部传入的 callbacks 合并进内部回调列表
        if callbacks is not None:
            if isinstance(callbacks, list):
                cb_list.extend(callbacks)
            else:
                cb_list.append(callbacks)
        
        validation_data = (x_val, y_val) if x_val is not None else None
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=cb_list,
            shuffle=False,  # 保持时间序列顺序，避免数据泄漏
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
        """加载模型

        如果模型文件包含匿名 lambda（旧版本保存），Keras 默认拒绝反序列化。
        在捕获到相应的 ValueError 时，作为兼容回退启用不安全反序列化并重试一次。
        注意：仅在你信任本地模型文件时才允许该回退。
        """
        # 处理 Windows 中文路径问题
        safe_path = _get_safe_path(filepath)
        custom_objs = {
            "SliceLayer": SliceLayer, 
            "ReduceSumLayer": ReduceSumLayer,
        }
        try:
            # 使用 compile=False 避免尝试恢复训练配置，兼容不同 TF/Keras 版本
            # LSTM 兼容性已通过 monkey patch 方式全局处理
            self.model = keras.models.load_model(safe_path, compile=False, custom_objects=custom_objs)
        except ValueError as e:
            msg = str(e)
            if 'Lambda' in msg or 'deserialization of a `Lambda`' in msg:
                logger.warning("检测到 Lambda 层反序列化限制，尝试启用不安全反序列化并重试加载模型（仅在本地文件可信时可用）")
                # 首选：尝试启用不安全反序列化（若可用）并重试
                tried = False
                try:
                    keras.config.enable_unsafe_deserialization()
                    tried = True
                except Exception:
                    tried = False

                # 重试加载；部分 Keras 版本支持 safe_mode 参数可以直接关闭保护
                try:
                    self.model = keras.models.load_model(safe_path, compile=False, custom_objects=custom_objs)
                except Exception:
                    # 如果上面仍失败，尝试使用 safe_mode=False（某些版本的 load_model 支持该参数）
                    try:
                        self.model = keras.models.load_model(safe_path, compile=False, safe_mode=False, custom_objects=custom_objs)
                    except TypeError:
                        # load_model 不支持 safe_mode 参数 -> 无更多回退方法
                        raise
            else:
                raise
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
            ball_input = SliceLayer(i, name=f"slice_{i}")(inputs)
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

        # 为每个输出显式指定 metrics，确保每个 metric 名称唯一，避免 Keras 在多输出时命名冲突
        metrics_map = {}
        for i in range(self.num_balls):
            out_name = f"ball_{i}_output"
            # 使用 SparseCategoricalAccuracy，给每个输出一个唯一的 metric 名称
            metrics_map[out_name] = [tf.keras.metrics.SparseCategoricalAccuracy(name=f"{out_name}_acc")]

        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics_map
        )
        logger.info("多输出模型编译完成")
    
    def train(self, x_train, y_train_list, x_val=None, y_val_list=None, 
              epochs=50, batch_size=32, early_stopping_patience=10, callbacks=None):
        """训练模型"""
        if self.model is None:
            self.compile_model()
        
        cb_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        # 合并外部回调
        if callbacks is not None:
            if isinstance(callbacks, list):
                cb_list.extend(callbacks)
            else:
                cb_list.append(callbacks)

        validation_data = (x_val, y_val_list) if x_val is not None else None
        
        self.history = self.model.fit(
            x_train, y_train_list,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=cb_list,
            shuffle=False,  # 保持时间序列顺序，避免数据泄漏
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
        """加载模型（含兼容性回退）"""
        # 处理 Windows 中文路径问题
        safe_path = _get_safe_path(filepath)
        custom_objs = {
            "SliceLayer": SliceLayer, 
            "ReduceSumLayer": ReduceSumLayer,
        }
        try:
            self.model = keras.models.load_model(safe_path, compile=False, custom_objects=custom_objs)
        except ValueError as e:
            msg = str(e)
            if 'Lambda' in msg or 'deserialization of a `Lambda`' in msg:
                logger.warning("检测到 Lambda 层反序列化限制（蓝球模型），尝试启用不安全反序列化并重试加载模型")
                try:
                    keras.config.enable_unsafe_deserialization()
                except Exception:
                    pass
                try:
                    self.model = keras.models.load_model(safe_path, compile=False, custom_objects=custom_objs)
                except Exception:
                    try:
                        self.model = keras.models.load_model(safe_path, compile=False, safe_mode=False, custom_objects=custom_objs)
                    except TypeError:
                        raise
            else:
                raise
        logger.info(f"多输出模型已加载: {safe_path}")

    def summary(self):
        """打印模型摘要"""
        if self.model is None:
            logger.warning("模型未初始化")
            return
        self.model.summary()


class MultiLabelLSTM:
    """多标签 LSTM 模型 - 对每个可能的号码输出独立的概率 (sigmoid)

    适用场景：红球作为集合（无序、多选），输出长度为 n_class，使用 sigmoid 激活和
    binary_crossentropy 训练。预测时可直接取 Top-K 或按阈值选取。
    """

    def __init__(self, n_class, window_size, num_balls, embedding_size=64,
                 hidden_size=128, num_layers=2, dropout_rate=0.2,
                 use_numeric_features=False, num_numeric_features=0, bidirectional=False,
                 use_attention=False):
        self.n_class = n_class
        self.window_size = window_size
        self.num_balls = num_balls
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_numeric_features = use_numeric_features
        self.num_numeric_features = num_numeric_features
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.model = None
        self.history = None

    def build_model(self):
        logger.info("构建多标签模型 (MultiLabelLSTM) ...")
        inputs = keras.Input(shape=(self.window_size, self.num_balls), dtype=tf.int32, name="inputs")

        numeric_input = None
        if self.use_numeric_features:
            numeric_input = keras.Input(shape=(self.num_numeric_features,), dtype=tf.float32, name="numeric_input")

        embeddings = []
        for i in range(self.num_balls):
            ball_input = SliceLayer(i, name=f"ml_slice_{i}")(inputs)
            emb = layers.Embedding(
                input_dim=self.n_class,
                output_dim=self.embedding_size,
                name=f"ml_embedding_{i}"
            )(ball_input)
            embeddings.append(emb)

        if len(embeddings) > 1:
            x = layers.Concatenate()(embeddings)
        else:
            x = embeddings[0]

        x = layers.Dropout(self.dropout_rate)(x)

        for i in range(self.num_layers):
            lstm_layer = layers.LSTM(
                units=self.hidden_size,
                # 如果最后一层后会使用 attention，则需要输出所有时间步
                return_sequences=(i < self.num_layers - 1) or (self.use_attention and i == self.num_layers - 1),
                dropout=self.dropout_rate,
                name=f"ml_lstm_{i}"
            )
            if self.bidirectional:
                x = layers.Bidirectional(lstm_layer, name=f"bidirectional_ml_lstm_{i}")(x)
            else:
                x = lstm_layer(x)

        # 如果使用注意力机制并且最后一层返回序列，执行注意力权重聚合
        if self.use_attention:
            # x shape: (batch, time, features)
            # 计算注意力得分
            scores = layers.Dense(1, activation='tanh', name='attn_score')(x)
            scores = layers.Flatten()(scores)
            weights = layers.Activation('softmax', name='attn_weights')(scores)
            weights = layers.RepeatVector(x.shape[-1])(weights)
            weights = layers.Permute([2, 1])(weights)
            x = layers.Multiply()([x, weights])
            x = ReduceSumLayer(axis=1, name='attn_context')(x)

        # 如果使用数值特征，将其投影并与 LSTM 输出拼接
        if self.use_numeric_features and numeric_input is not None:
            proj = layers.Dense(units=max(32, self.num_numeric_features // 2), activation='relu', name='numeric_proj')(numeric_input)
            x = layers.Concatenate(name='concat_numeric')([x, proj])

        outputs = layers.Dense(units=self.n_class, activation='sigmoid', name="multi_label_output")(x)

        if self.use_numeric_features and numeric_input is not None:
            self.model = Model(inputs=[inputs, numeric_input], outputs=outputs, name="MultiLabelLSTM")
        else:
            self.model = Model(inputs=inputs, outputs=outputs, name="MultiLabelLSTM")
        logger.info("多标签模型构建完成")
        return self.model

    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            self.build_model()
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')]
        )
        logger.info("多标签模型编译完成")

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=32,
              early_stopping_patience=10, callbacks=None):
        if self.model is None:
            self.compile_model()

        cb_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        if callbacks is not None:
            if isinstance(callbacks, list):
                cb_list.extend(callbacks)
            else:
                cb_list.append(callbacks)

        validation_data = (x_val, y_val) if x_val is not None else None

        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=cb_list,
            shuffle=False,  # 保持时间序列顺序，避免数据泄漏
            verbose=1
        )
        logger.info("多标签模型训练完成")
        return self.history

    def predict_proba(self, x):
        if self.model is None:
            raise ValueError("模型未初始化！")
        return self.model.predict(x, verbose=0)

    def predict(self, x, top_k=6):
        proba = self.predict_proba(x)
        if top_k is None:
            # 返回概率阈值选取 (默认 0.5)
            return (proba >= 0.5).astype(int)
        # 返回 Top-k 索引（1-indexed）
        top_idx = np.argsort(-proba, axis=1)[:, :top_k] + 1
        return top_idx

    def save(self, filepath):
        if self.model is None:
            raise ValueError("模型未初始化！")
        self.model.save(filepath)
        logger.info(f"多标签模型已保存到: {filepath}")

    def load(self, filepath):
        # 处理 Windows 中文路径问题
        safe_path = _get_safe_path(filepath)
        custom_objs = {
            "SliceLayer": SliceLayer, 
            "ReduceSumLayer": ReduceSumLayer,
        }
        try:
            self.model = keras.models.load_model(safe_path, compile=False, custom_objects=custom_objs)
        except ValueError as e:
            msg = str(e)
            if 'Lambda' in msg or 'deserialization of a `Lambda`' in msg:
                logger.warning("检测到 Lambda 层反序列化限制（红球多标签模型），尝试启用不安全反序列化并重试加载模型")
                try:
                    keras.config.enable_unsafe_deserialization()
                except Exception:
                    pass
                try:
                    self.model = keras.models.load_model(safe_path, compile=False, custom_objects=custom_objs)
                except Exception:
                    try:
                        self.model = keras.models.load_model(safe_path, compile=False, safe_mode=False, custom_objects=custom_objs)
                    except TypeError:
                        raise
            else:
                raise
        logger.info(f"多标签模型已加载: {safe_path}")

    def summary(self):
        if self.model is None:
            logger.warning("模型未初始化")
            return
        self.model.summary()
