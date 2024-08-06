import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch

def create_sliding_window(data, n_steps):
    X, y = [], []
    # feature_cols = ['ForceX', 'ForceY', 'ForceZ']
    feature_cols = ['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ'] # Feature column names
    for i in range(len(data) - n_steps):
        # Determine the end of the current sliding window
        end_ix = i + n_steps
        # Collect inputs and outputs
        seq_x = data.iloc[i:end_ix][feature_cols]  # Selects only the columns for features
        seq_y = data.iloc[end_ix, data.columns.get_loc('Index')]-1 # Assumes 'Index' column for output
        X.append(seq_x.values.flatten())  # Flatten the array to create a single feature vector per window
        y.append(seq_y)
    return np.array(X), np.array(y)

def data_Pro_window(File, n_steps):
    data = pd.read_csv(File)

    # Removing rows where Index is 0
    data = data[data['Index'] != 0]

    # Creating sliding window data
    X, y = create_sliding_window(data, n_steps)

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Converting numpy arrays to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Creating DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader

def data_Pro(File):
    data = pd.read_csv(File)

    # Removing rows where Index is 0
    data = data[data['Index'] != 0]

    # Defining features and target variable
    feature_cols = ['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']
    X = data[feature_cols].values
    y = (data['Index'] - 1).values  # Adjusting labels to start from 0

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Converting numpy arrays to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Creating DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def interpolate_csv(input_csv, output_csv, sampling_rate):
    """
    使用线性插值法将CSV文件处理为固定采样率的数据。

    Args:
        input_csv (str): 输入CSV文件的路径。
        output_csv (str): 输出CSV文件的路径。
        sampling_rate (float): 目标采样率，以Hz为单位。
    """

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # # 将时间戳转换为秒为单位的浮点数
    # df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype('int64') / 1

    # 获取开始和结束时间
    start_time = df['Timestamp'].iloc[0]
    end_time = df['Timestamp'].iloc[-1]

    # 生成新的时间戳序列
    new_timestamps = pd.Series(
        data=np.arange(start_time, end_time, 1 / sampling_rate),
        name='Timestamp'
    )

    # 使用新的时间戳序列创建新的DataFrame
    new_df = pd.DataFrame(new_timestamps)

    # 将原始数据合并到新的DataFrame中
    new_df = pd.merge_asof(new_df, df, on='Timestamp', direction='forward')

    # 对除时间戳以外的所有列进行线性插值
    for column in df.columns[1:]:
        new_df[column] = new_df[column].interpolate(method='linear')

    # 保存插值后的数据到新的CSV文件
    new_df.to_csv(output_csv, index=False)

def interpolate_csv_by_index(input_csv, output_csv, sampling_rate):
    """
    对每个索引值执行线性插值，并将CSV文件处理为固定采样率的数据。

    Args:
        input_csv (str): 输入CSV文件的路径。
        output_csv (str): 输出CSV文件的路径。
        sampling_rate (float): 目标采样率，以Hz为单位。
    """

    # 计算重采样的时间间隔
    resample_interval = f"{1 / sampling_rate}S"
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    # 按索引分组
    grouped = df.groupby("Index")
    # 创建一个空列表来存储插值后的数据帧
    interpolated_dfs = []

    # 迭代每个组（即每个唯一的索引）
    for index, group in grouped:
        # 仅在当前组内执行插值
        interpolated_group = group.interpolate(method='linear', axis=0)
        # 将插值后的组追加到列表中
        interpolated_dfs.append(interpolated_group)

    # 将插值后的数据帧连接在一起
    interpolated_df = pd.concat(interpolated_dfs)
    # 保存插值后的数据到新的CSV文件
    interpolated_df.to_csv(output_csv, index=False)

def interpolate_csv_by_index_with_average(input_csv, output_csv, sampling_rate, window_size=5):
    """
    对每个索引值执行基于窗口平均的线性插值，并将CSV文件处理为固定采样率的数据。

    Args:
        input_csv (str): 输入CSV文件的路径。
        output_csv (str): 输出CSV文件的路径。
        sampling_rate (float): 目标采样率，以Hz为单位。
        window_size (int): 用于计算滑动平均值的窗口大小。
    """

    # 计算重采样的时间间隔
    resample_interval = f"{1 / sampling_rate}s"  # 使用 's' 而不是 'S'

    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 将 "Timestamp" 列转换为 DatetimeIndex
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit='s')  # 假设时间戳单位是秒

    # 按索引分组
    grouped = df.groupby("Index")

    # 创建一个空列表来存储插值后的数据帧
    interpolated_dfs = []

    # 迭代每个组（即每个唯一的索引）
    for index, group in grouped:
        # 提取时间戳
        timestamps = group["Timestamp"]

        # 计算滑动窗口平均值
        for column in group.columns:
            if column != "Index" and column != "Timestamp":
                group[column] = (
                    group[column].rolling(window=window_size, center=True).mean()
                )

        # 重采样数据 (保留时间戳)
        group = group.resample(resample_interval, on="Timestamp").mean()

        # 执行线性插值
        interpolated_group = group.interpolate(method="linear", axis=0)

        # 使用原始时间戳覆盖插值后的时间戳
        interpolated_group["Timestamp"] = timestamps

        # 将插值后的组追加到列表中
        interpolated_dfs.append(interpolated_group)

    # 将插值后的数据帧连接在一起
    interpolated_df = pd.concat(interpolated_dfs)

    interpolated_df = interpolated_df[interpolated_df["Resultant Force"].notna()]

    # 保存插值后的数据到新的CSV文件
    interpolated_df.to_csv(output_csv, index=False)

def interpolate_csv_by_rolling_mean(input_csv, output_csv, window_size):
    """
    使用滚动窗口的均值对CSV文件进行插值。

    Args:
        input_csv (str): 输入CSV文件的路径。
        output_csv (str): 输出CSV文件的路径。
        window_size (int): 滚动窗口的大小。
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 按索引分组
    grouped = df.groupby("Index")

    # 创建一个空列表来存储插值后的数据帧
    interpolated_dfs = []

    # 迭代每个组（即每个唯一的索引）
    for index, group in grouped:
        # 使用滚动窗口方法对每个组进行插值
        interpolated_group = group.fillna(group.rolling(window=window_size, min_periods=1).mean())
        # 将插值后的组追加到列表中
        interpolated_dfs.append(interpolated_group)

    # 将插值后的数据帧连接在一起
    interpolated_df = pd.concat(interpolated_dfs)

    # 保存插值后的数据到新的CSV文件
    interpolated_df.to_csv(output_csv, index=False)

def create_combined_train_loader(files, batch_size=32):
    """
    Combines data from multiple CSV files into a single training DataLoader.

    Args:
        files (list): List of CSV file paths.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the combined training data.
    """

    all_train_datasets = []
    for file in files:
        data = pd.read_csv(file)
        data = data[data['Index'] != 0]

        feature_cols = ['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']
        X = data[feature_cols].values
        y = (data['Index'] - 1).values

        X_train = torch.tensor(X, dtype=torch.float32)  # Using all data for training
        y_train = torch.tensor(y, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        all_train_datasets.append(train_dataset)

    # Combine all training datasets
    combined_train_dataset = ConcatDataset(all_train_datasets)

    # Create DataLoader for combined training data
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    return combined_train_loader

# dic = 'H:/data/240327ft/'
# out='interpolate_'
# dataname1 = 'ft2.csv'
# dataname2 = 'ft4.csv'
j=3
dic='H:\\data\\brushFTarea\\240703\\'
dic1='H:\\data\\brushFTarea\\electric\\'
dataname1 = 'ft1_processed.csv'
dataname2 = 'ft2_processed.csv'
dataname3 = 'ft3_processed.csv'
dataname4 = 'ft3_processed.csv'
File=[]
for k in range(j):
    File.append(dic+'ft{}_processed.csv'.format(k+1))

File1 = dic1 + dataname1
File2 = dic1 + dataname2
File3 = dic1 + dataname3


n_steps=1

dic2 = 'H:\\data\\brushFTarea\\softelectric\\'

file_paths=[]
j=6
for k in range(j):
    file_paths.append(dic2+'ft{}_interpolated.csv'.format(k+1))



# train,test=data_Pro(File1)
# train2,test2=data_Pro(File2)
# train2,test2=data_Pro(File2)
#
# train1,test1=data_Pro(File1)
# train2,test1=data_Pro(File2)

# train2,test2=data_Pro(File2,n_steps)

# train_comb = create_combined_train_loader(File[:-1])
# train=train_comb
# test=create_combined_train_loader([File[-1]])

train_comb = create_combined_train_loader(file_paths[:-1])
# train_comb = create_combined_train_loader(file_paths)
train=train_comb

test=create_combined_train_loader([file_paths[-1]])
# test=create_combined_train_loader(["H:\\data\\brushFTarea\\tester\\ft11_interpolated.csv"])
