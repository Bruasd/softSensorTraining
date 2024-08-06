import time
import numpy as np
from scipy.signal import butter, lfilter

# 设置参数
sampling_rate = 1000  # 采样率 (Hz)
cutoff_frequency = 50  # 截止频率 (Hz)
filter_order = 4  # 滤波器阶数

# 设计巴特沃斯低通滤波器
butter_b, butter_a = butter(filter_order, cutoff_frequency / (sampling_rate / 2), 'low')
print(butter_b)
print(butter_a)

# 初始化数据采集
# ...

# 主循环
# while True:
#     # 读取力传感器数据
#     force_data = acquire_data()
#
#     # 低通滤波
#     filtered_data = lfilter(b, a, force_data)
#
#     # 数据预处理
#     # ...
#
#     # 神经网络训练
#     # ...
#
#     # 控制输出或其他操作
#     # ...
#
#     # 延时
#     time.sleep(1/sampling_rate)