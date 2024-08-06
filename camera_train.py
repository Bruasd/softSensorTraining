import pandas as pd
# from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义模型
class MLP(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.model(x)

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(6, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 6),
#         )
#
#     def forward(self, x):
#         return self.model(x)

# 读取数据,并进行数据预处理
def loadData(willSplit, testSplit, forceFile):
    all_data = []

    for file_path in forceFile:
        force_data = pd.read_csv(file_path, usecols=['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ'])
        position_data = pd.read_csv(file_path, usecols=['Alpha', 'Beta', 'Gama', 'X', 'Y', 'Z'])
        data = pd.concat([position_data, force_data], axis=1)
        all_data.append(data)
    all_data = pd.concat(all_data, ignore_index=True)

    # 分成输入和输出
    X = all_data.iloc[:, 0:6].values
    y = all_data.iloc[:, 6:12].values

    # 分成训练集和测试集
    if willSplit:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSplit, random_state=1)

        # X_train = X_train.values
        # y_train = y_train.values
        # X_test = X_test.values
        # y_test = y_test.values
        return X_train, X_test, y_train, y_test

    if not willSplit:
        return X, y

# 训练模型
def train_model(X_train, y_train, epochs, model_name: str = 'myModel.pth'):
    model.to(device)
    X_train = Variable(torch.from_numpy(X_train).float()).to(device)
    y_train = Variable(torch.from_numpy(y_train).float()).to(device)
    loss_list = []

    for epoch in range(epochs):
        outputs = model(X_train)
        loss = loss_func(outputs, y_train)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播 对模型的损失值进行反向传播，计算模型中每个可学习参数的梯度。
        optimizer.step()  # 根据之前计算的梯度，通过优化器来更新模型的参数，使得模型朝着更优的方向优化。
        loss_list.append(loss.item())
        if epoch % 10 == 0:
            print('Epoch:', epoch, 'Loss:', loss.item())
            # 保存模型
    torch.save(model.state_dict(), model_name)
    print('train model successfully!')
    plt.plot(range(epochs), loss_list, color='blue', linewidth=1.5, linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()


# 预测
def predict_model(model_name, X_test, y_test):
    model.load_state_dict(torch.load(model_name))  # 读取模型
    model.eval()
    model.to(device)
    print('test data: ', len(X_test))

    input = torch.from_numpy(X_test).float().to(device)
    pred = model(input).detach().cpu().numpy()

    # 评估
    print('MSE:', mean_squared_error(y_test, pred))
    print('MAE:', mean_absolute_error(y_test, pred))
    print('R2:', r2_score(y_test, pred))

    print('Fx MSE:', mean_squared_error(y_test[:, 0], pred[:, 0]))
    print('Fx MAE:', mean_absolute_error(y_test[:, 0], pred[:, 0]))
    print('Fx R2:', r2_score(y_test[:, 0], pred[:, 0]))

    print('Fy MSE:', mean_squared_error(y_test[:, 1], pred[:, 1]))
    print('Fy MAE:', mean_absolute_error(y_test[:, 1], pred[:, 1]))
    print('Fy R2:', r2_score(y_test[:, 1], pred[:, 1]))

    print('Fz MSE:', mean_squared_error(y_test[:, 2], pred[:, 2]))
    print('Fz MAE:', mean_absolute_error(y_test[:, 2], pred[:, 2]))
    print('Fz R2:', r2_score(y_test[:, 2], pred[:, 2]))

    print('Tx MSE:', mean_squared_error(y_test[:, 3], pred[:, 3]))
    print('Tx MAE:', mean_absolute_error(y_test[:, 3], pred[:, 3]))
    print('Tx R2:', r2_score(y_test[:, 3], pred[:, 3]))

    print('Ty MSE:', mean_squared_error(y_test[:, 4], pred[:, 4]))
    print('Ty MAE:', mean_absolute_error(y_test[:, 4], pred[:, 4]))
    print('Ty R2:', r2_score(y_test[:, 4], pred[:, 4]))

    print('Tz MSE:', mean_squared_error(y_test[:, 5], pred[:, 5]))
    print('Tz MAE:', mean_absolute_error(y_test[:, 5], pred[:, 5]))
    print('Tz R2:', r2_score(y_test[:, 5], pred[:, 5]))

    # 画图
    fig, axes = plt.subplots(6, 1, figsize=(8, 12), constrained_layout=True)

    axes[0].plot(range(y_test.shape[0]), y_test[:, 0], linewidth=1.5, linestyle='-', label='True')
    axes[0].plot(range(y_test.shape[0]), pred[:, 0], linewidth=1, linestyle='-.', label='Predicted')
    axes[0].set_xticks([])
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel(r'$F_{x}(N)$')
    axes[0].yaxis.tick_left()
    axes[0].yaxis.set_label_position('left')
    # axes[0].set_title('X Force')

    axes[1].plot(range(y_test.shape[0]), y_test[:, 1], linewidth=1.5, linestyle='-')
    axes[1].plot(range(y_test.shape[0]), pred[:, 1], linewidth=1, linestyle='-.')
    # axes[1].legend(loc='upper right')
    axes[1].set_xticks([])

    axes[1].set_ylabel(r'$F_{y}(N)$')
    axes[1].yaxis.tick_left()
    axes[1].yaxis.set_label_position('left')

    axes[2].plot(range(y_test.shape[0]), y_test[:, 2], linewidth=1.5, linestyle='-')
    axes[2].plot(range(y_test.shape[0]), pred[:, 2], linewidth=1, linestyle='-.')
    # axes[2].legend(loc='upper right')
    axes[2].set_xticks([])
    axes[2].set_ylabel(r'$F_{z}(N)$')
    axes[2].yaxis.set_label_position('left')
    # axes[2].set_title('z Force')

    axes[3].plot(range(y_test.shape[0]), y_test[:, 3], linewidth=1.5, linestyle='-')
    axes[3].plot(range(y_test.shape[0]), pred[:, 3], linewidth=1, linestyle='-.')
    # axes[3].legend(loc='upper right')
    axes[3].set_xticks([])
    axes[3].set_ylabel(r'$T_{x}(Nm)$')
    axes[3].yaxis.set_label_position('left')

    axes[4].plot(range(y_test.shape[0]), y_test[:, 4], linewidth=1.5, linestyle='-')
    axes[4].plot(range(y_test.shape[0]), pred[:, 4], linewidth=1, linestyle='-.')
    # axes[4].legend(loc='upper right')
    axes[4].set_xticks([])
    axes[4].set_ylabel(r'$T_{y}(Nm)$')
    axes[4].yaxis.set_label_position('left')

    axes[5].plot(range(y_test.shape[0]), y_test[:, 5], linewidth=1.5, linestyle='-')
    axes[5].plot(range(y_test.shape[0]), pred[:, 5], linewidth=1, linestyle='-.')
    axes[5].legend(loc='upper right')
    axes[5].set_ylabel(r'$T_{z}(Nm)$')
    axes[5].yaxis.set_label_position('left')
    axes[5].set_xlabel('The number of test samples')
    # 调整子图之间的距离
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.08, 0.5)
    # 使用tight_layout确保布局合适
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 定义用于训练和测试的数据
    trainnumber=[5,6]
    testnumber=[6]

    trainingFile=[]
    for number in trainnumber:
        trainingFile.append('H:\\data\\softSensor\\merged{}.csv'.format(number))
    testFile=[]
    for number in testnumber:
        testFile.append('H:\\data\\softSensor\\merged{}.csv'.format(number))
    # 读取数据，划分测试集和训练集
    data_train, data_test, value_train, value_test = loadData(willSplit=True, testSplit=0.1,
                                                              forceFile=trainingFile)

    # 对数据进行标准化处理
    scaler = StandardScaler()
    scaler.fit(data_train)  # 用于计算均值和方差
    data_train = scaler.transform(data_train)  # 标准化训练集
    data_test = scaler.transform(data_test)  # 标准化测试集

    # 保存 scaler 对象到文件
    filename = 'H:\\data\\softSensor\\'+'my_scaler.pkl'  # 选择文件名
    pickle.dump(scaler, open(filename, 'wb'))

    # 初始化模型、损失函数、优化器
    model = MLP().to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch = 150
    model_name = f'mlpModel{epoch}.pth'
    train_model(data_train, value_train, epochs=epoch, model_name='H:\\data\\softSensor\\'+model_name)
    predict_model(model_name='H:\\data\\softSensor\\'+model_name, X_test=data_test, y_test=value_test)

    # 读取一份其他文件用做测试，不分割文件
    testData, testValue = loadData(willSplit=False, testSplit=0.1, forceFile=testFile)

    testData = scaler.transform(testData)
    predict_model(model_name=model_name, X_test=testData, y_test=testValue)