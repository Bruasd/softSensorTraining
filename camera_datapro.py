import pandas as pd



def mean_centering(file_path, output_file_path):
    """
    读取并处理CSV文件，对Index为0的数据计算Alpha、Beta、Gama、X、Y、Z的均值，
    并将所有数据减去对应的均值，使其从0开始。

    参数:
    file_path (str): CSV文件路径

    返回:
    pd.DataFrame: 处理后的数据
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 计算Index为0时的均值
    index_zero_means = df[df['Index'] == 0][['Alpha', 'Beta', 'Gama', 'X', 'Y', 'Z']].mean()

    # 将所有数据减去均值
    df[['Alpha', 'Beta', 'Gama', 'X', 'Y', 'Z']] -= index_zero_means

    df[['X', 'Y', 'Z']] *= 10000

    df.to_csv(output_file_path, index=False)


def merge_sensor_data( pose_file_path, ft_file_path,output_file_path, tolerance=0.05):
    """
    合并FT传感器数据和动捕设备数据，并保存到新的CSV文件。

    参数:
    ft_file_path (str): FT传感器数据文件路径
    motion_file_path (str): 动捕设备数据文件路径
    output_file_path (str): 输出合并后数据文件路径
    tolerance (float): 时间戳合并的容差，单位为秒，默认为0.05

    返回:
    pd.DataFrame: 合并后的数据
    """
    # 读取FT传感器数据
    FT_data = pd.read_csv(ft_file_path)

    # 读取摄像头数据
    motion_data = pd.read_csv(pose_file_path)

    # 确保时间戳列是已排序的
    motion_data.sort_values('Timestamp', inplace=True)
    FT_data.sort_values('Timestamp', inplace=True)

    # 使用merge_asof方法按照时间戳对齐合并数据
    merged_data = pd.merge_asof(motion_data,FT_data, on='Timestamp', tolerance=tolerance)
    merged_data.dropna(inplace=True)

    # 填充缺失的数据，使用ffill方法进行前向填充
    # merged_data.ffill(inplace=True)

    # 保存合并后的数据为新的文件
    merged_data.to_csv(output_file_path, index=False)


def mean_filter(input_file, output_file, window_size=9):
    """
    对 CSV 文件中的数据应用均值滤波，并忽略边缘数据。

    参数:
    input_file (str): 输入 CSV 文件路径。
    output_file (str): 输出 CSV 文件路径。
    window_size (int): 滤波窗口大小（默认为 3）。
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 定义应用均值滤波的内部函数
    def apply_mean_filter(group, window_size):
        # 应用均值滤波
        filtered_group = group.rolling(window=window_size, center=True).mean()
        # 将边缘数据设置为 NaN
        half_window = window_size // 2
        filtered_group.iloc[:half_window] = float('nan')
        filtered_group.iloc[-half_window:] = float('nan')
        return filtered_group

    # 对每个 Index 内的数据应用均值滤波，并忽略边缘数据
    filtered_df = df.groupby('Index').apply(lambda group: group.apply(
        lambda col: apply_mean_filter(col, window_size) if col.name not in ['Timestamp', 'Index'] else col))

    # 处理分组结果
    filtered_df.reset_index(drop=True, inplace=True)

    # 去除全0行
    filtered_df = filtered_df.loc[~(filtered_df.drop(columns=['Timestamp', 'Index']) == 0).all(axis=1)]

    # 保存结果到新的 CSV 文件
    filtered_df.to_csv(output_file, index=False)

    print(f"均值滤波处理完毕，结果已保存到 '{output_file}'")


def merge_sensor_data_with_averaging( pose_file_path, ft_file_path,output_file_path, tolerance=0.05, window_size=5):
    """
    合并FT传感器数据和动捕设备数据，并保存到新的CSV文件。

    对FT数据进行局部时间窗口平均以匹配motion数据的时间戳。

    参数:
    ft_file_path (str): FT传感器数据文件路径
    pose_file_path (str): 动捕设备数据文件路径
    output_file_path (str): 输出合并后数据文件路径
    tolerance (float): 时间戳合并的容差，单位为秒，默认为0.05
    window_size (int): 用于平均FT数据的窗口大小（前后帧数），默认为5

    返回:
    pd.DataFrame: 合并后的数据
    """

    # 读取FT传感器数据
    FT_data = pd.read_csv(ft_file_path)

    # 读取动捕设备数据
    motion_data = pd.read_csv(pose_file_path)

    # 确保时间戳列是已排序的
    motion_data.sort_values('Timestamp', inplace=True)
    FT_data.sort_values('Timestamp', inplace=True)

    # 创建一个新的DataFrame存储合并后的数据
    merged_data = pd.DataFrame()

    for i in range(len(motion_data)):
        # 获取当前motion数据的时间戳
        current_timestamp = motion_data['Timestamp'].iloc[i]
        print(i)

        # 找到FT数据中与当前motion数据时间戳最接近的索引
        closest_idx = (FT_data['Timestamp'] - current_timestamp).abs().argmin()

        # 计算用于平均FT数据的起始和结束索引
        start_idx = max(0, closest_idx - window_size)
        end_idx = min(len(FT_data), closest_idx + window_size + 1)

        # 获取对应时间窗口内的FT数据并计算均值
        averaged_ft_data = FT_data.iloc[start_idx:end_idx].mean()

        # 将平均后的FT数据与当前motion数据合并
        merged_row = pd.concat([averaged_ft_data, motion_data.iloc[[i]]], axis=1)
        merged_data = pd.concat([merged_data, merged_row], ignore_index=True)

    # 保存合并后的数据为新的文件
    merged_data.to_csv(output_file_path, index=False)

for i in range(4):

    # dic="H:\\data\\softSensor\\"
    dic = "H:\\data\\brushSoftSensor\\"
    FileP= dic+"pose{}.csv".format(i+1)
    OutFileP= dic+"pose{}_mean_centered.csv".format(i+1)
    InterFile=dic+"ft{}_interpolated.csv".format(i+1)
    FileF = dic + "ft{}.csv".format(i + 1)
    OutFileF = dic + "ft{}_mean_filtered.csv".format(i + 1)
    MergedFile=dic+"merged{}.csv".format(i + 1)
    MergedFile1 = dic + "merged1{}.csv".format(i + 1)
    mean_centering(FileP, OutFileP)
    mean_filter(FileF,OutFileF,window_size=9)
    # merge_sensor_data(OutFileP, FileF, MergedFile)
    merge_sensor_data(OutFileP, OutFileF, MergedFile)


