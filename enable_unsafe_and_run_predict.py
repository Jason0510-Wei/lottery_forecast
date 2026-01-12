# -*- coding:utf-8 -*-
"""
小脚本：在导入模型前启用 Keras 不安全反序列化（仅在你信任本地模型时使用）
用法：
  python .\enable_unsafe_and_run_predict.py --name ssq

注意：启用不安全反序列化存在安全风险，确保你信任仓库中的模型文件。
"""
from tensorflow import keras
import sys

# 启用不安全反序列化以兼容旧模型（有 lambda 的 .h5）
try:
    keras.config.enable_unsafe_deserialization()
    print("已启用 Keras 不安全反序列化（仅本次进程）。")
except Exception as e:
    print("启用不安全反序列化失败（可能 Keras 版本不支持），错误：", e)

# 导入并运行主预测脚本（它会在导入时解析命令行参数）
import run_predict  # run_predict 模块已在导入时解析了 argv

# run_predict 的主函数在模块底部通过 `if __name__ == '__main__'` 保护，
# 因此导入时并不会自动执行 main()。这里显式调用以在同一进程中运行预测。
try:
  run_predict.main()
except Exception as e:
  print("运行 run_predict 时发生错误：", e)
