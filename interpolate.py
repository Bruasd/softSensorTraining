import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from removezero import removezero
from datapro import interpolate_csv_by_index_with_average,interpolate_csv_by_rolling_mean


def interpolate_csv_by_index(input_csv, output_csv, sampling_rate):
    """
    按Index分组并对时间戳进行插值，生成新的时间戳和相应的力和力矩数据，最终写入新的CSV文件。

    参数：
    input_csv (str): 输入CSV文件路径。
    output_csv (str): 输出CSV文件路径。
    sampling_rate (float): 采样率（Hz），如100Hz。
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 定义插值分辨率
    step = 1 / sampling_rate  # 时间间隔

    # 创建一个空的DataFrame来存储插值后的数据
    interpolated_data = pd.DataFrame()

    # 根据Index分组
    groups = df.groupby('Index')

    # 用于记录Index的顺序
    index_order = []

    for name, group in groups:
        # 记录Index的顺序
        index_order.append(name)

        # 获取当前组的时间戳
        timestamps = group['Timestamp'].values

        # 生成新的时间戳
        new_timestamps = np.arange(timestamps.min(), timestamps.max(), step)

        # 对每个力和力矩数据进行线性插值
        new_data = {
            'Timestamp': new_timestamps,
            'ForceX': np.interp(new_timestamps, timestamps, group['ForceX']),
            'ForceY': np.interp(new_timestamps, timestamps, group['ForceY']),
            'ForceZ': np.interp(new_timestamps, timestamps, group['ForceZ']),
            'TorqueX': np.interp(new_timestamps, timestamps, group['TorqueX']),
            'TorqueY': np.interp(new_timestamps, timestamps, group['TorqueY']),
            'TorqueZ': np.interp(new_timestamps, timestamps, group['TorqueZ']),
            'Index': name  # 保留索引
        }

        # 将新数据转换为DataFrame
        interpolated_df = pd.DataFrame(new_data)

        # 计算新的合力
        interpolated_df['Resultant Force'] = np.sqrt(
            interpolated_df['ForceX']**2 +
            interpolated_df['ForceY']**2 +
            interpolated_df['ForceZ']**2
        )

        # 将插值后的数据添加到主DataFrame中
        interpolated_data = pd.concat([interpolated_df, interpolated_data])

    # 按照记录的Index顺序排序
    interpolated_data = interpolated_data.sort_values(by=['Index', 'Timestamp'])

    # 重置索引
    interpolated_data = interpolated_data.reset_index(drop=True)

    # 将插值后的数据写入新的CSV文件
    interpolated_data.to_csv(output_csv, index=False)

def interpolate_csv_by_index_window(input_csv, output_csv, sampling_rate, window_size=5):
    """
    按Index分组并对时间戳进行插值，生成新的时间戳和相应的力和力矩数据，最终写入新的CSV文件。
    使用前后若干帧的均值进行插值。

    参数：
    input_csv (str): 输入CSV文件路径。
    output_csv (str): 输出CSV文件路径。
    sampling_rate (float): 采样率（Hz），如100Hz。
    window_size (int): 用于计算均值的帧数，默认为5。
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 定义插值分辨率
    step = 1 / sampling_rate  # 时间间隔

    # 创建一个空的DataFrame来存储插值后的数据
    interpolated_data = pd.DataFrame()

    # 根据Index分组
    groups = df.groupby('Index')

    # 用于记录Index的顺序
    index_order = []

    for name, group in groups:
        # 记录Index的顺序
        index_order.append(name)

        # 获取当前组的时间戳
        timestamps = group['Timestamp'].values

        # 生成新的时间戳
        new_timestamps = np.arange(timestamps.min(), timestamps.max(), step)

        # 对每个力和力矩数据进行均值插值
        new_data = {
            'Timestamp': new_timestamps,
            'ForceX': np.interp(new_timestamps, timestamps, group['ForceX'].rolling(window_size, center=True).mean().bfill()),
            'ForceY': np.interp(new_timestamps, timestamps, group['ForceY'].rolling(window_size, center=True).mean().bfill()),
            'ForceZ': np.interp(new_timestamps, timestamps, group['ForceZ'].rolling(window_size, center=True).mean().bfill()),
            'TorqueX': np.interp(new_timestamps, timestamps, group['TorqueX'].rolling(window_size, center=True).mean().bfill()),
            'TorqueY': np.interp(new_timestamps, timestamps, group['TorqueY'].rolling(window_size, center=True).mean().bfill()),
            'TorqueZ': np.interp(new_timestamps, timestamps, group['TorqueZ'].rolling(window_size, center=True).mean().bfill()),
            'Index': name  # 保留索引
        }

        # 将新数据转换为DataFrame
        interpolated_df = pd.DataFrame(new_data)

        # 计算新的合力
        interpolated_df['Resultant Force'] = np.sqrt(
            interpolated_df['ForceX']**2 +
            interpolated_df['ForceY']**2 +
            interpolated_df['ForceZ']**2
        )

        # 将插值后的数据添加到主DataFrame中
        interpolated_data = pd.concat([interpolated_df, interpolated_data])

    # 按照记录的Index顺序排序
    interpolated_data = interpolated_data.sort_values(by=['Index', 'Timestamp'])

    # 删除 ForceX 列包含 NaN 的行
    interpolated_data.dropna(subset=['ForceX'], inplace=True)

    # 重置索引
    interpolated_data = interpolated_data.reset_index(drop=True)

    # 将插值后的数据写入新的CSV文件
    interpolated_data.to_csv(output_csv, index=False)


# File= "H:\\data\\brushFTarea\\240703\\ft{}.csv".format(i)
# OutFile= "H:\\data\\brushFTarea\\240703\\ft{}_processed.csv".format(i)


# File= "H:\\data\\brushFTarea\\electric\\ft{}.csv".format(i)
# OutFile= "H:\\data\\brushFTarea\\electric\\ft{}_processed.csv".format(i)
# InterFile="H:\\data\\brushFTarea\\electric\\ft{}_interpolated.csv".format(i)
for i in range(4):

    # dicsoftelec="H:\\data\\brushFTarea\\softelectric\\" #这是直接使用传感器测刷牙区域分类
    dicsoftelec = "H:\\data\\brushSoftSensor\\" #这是在传感器上放了个柔性传感器再测
    File= dicsoftelec+"ft{}.csv".format(i+1)
    OutFile= dicsoftelec+"ft{}_processed.csv".format(i+1)
    InterFile=dicsoftelec+"ft{}_interpolated.csv".format(i+1)
    print(InterFile)

    # threshold=0.4
    # removezero(File,OutFile,threshold)
    #
    # # Load the CSV file into a Pandas DataFrame
    # # data = pd.read_csv(OutFile)  # Replace "your_file.csv" with your actual file name
    #
    # sampling_rate = 100  # Hz
    # # 执行插值操作
    # # interpolate_csv_by_rolling_mean(OutFile, InterFile, 5)
    # interpolate_csv_by_index_window(OutFile, InterFile, sampling_rate,9)



    # Load the CSV file into a Pandas DataFrame
    # data = pd.read_csv(OutFile)  # Replace "your_file.csv" with your actual file name

    sampling_rate = 100  # Hz
    # 执行插值操作
    # interpolate_csv_by_rolling_mean(OutFile, InterFile, 5)
    interpolate_csv_by_index_window(File, OutFile, sampling_rate,9)

    threshold=0.5
    removezero(OutFile,InterFile,threshold)




    # j=11
    # dicsoftelec="H:\\data\\brushFTarea\\tester\\"
    # File= dicsoftelec+"ft{}.csv".format(j)
    # OutFile= dicsoftelec+"ft{}_processed.csv".format(j)
    # InterFile=dicsoftelec+"ft{}_interpolated.csv".format(j)
    # print(InterFile)
    #
    # threshold=0.4
    # removezero(File,OutFile,threshold)
    #
    # # Load the CSV file into a Pandas DataFrame
    # # data = pd.read_csv(OutFile)  # Replace "your_file.csv" with your actual file name
    #
    # sampling_rate = 100  # Hz
    # # 执行插值操作
    # # interpolate_csv_by_rolling_mean(OutFile, InterFile, 5)
    # interpolate_csv_by_index_window(OutFile, InterFile, sampling_rate,9)
