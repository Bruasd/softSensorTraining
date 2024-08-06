import pandas as pd
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
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=36, num_layers=3, output_size=6):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        h0, c0 = hidden  # 解包隐藏状态和细胞状态
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)

        # 解码隐藏状态
        out = self.fc(out[:, -1, :])
        return out

def loadData(willSplit, testSplit, forceFile, positionfile, seq_length):
    force_data = pd.read_csv(forceFile, usecols=['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ'])
    position_data = pd.read_csv(positionfile, usecols=['Alpha','Beta','Gama','X','Y','Z'])

    # 合并数据
    data = pd.concat([position_data,force_data], axis=1)

    # 构建输入序列和输出序列
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length, 0:6].values)  # 取 position 数据作为输入序列
        y.append(data.iloc[i+seq_length, 6:12].values)   # 取 force 数据作为输出序列

    X = np.array(X)
    y = np.array(y)

    # 分成训练集和测试集
    if willSplit:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSplit, random_state=1)
        return X_train, X_test, y_train, y_test

    if not willSplit:
        return X, y

def train_lstm_model(X_train, y_train, epochs, model, optimizer, loss_func, model_name='myModel.pth'):
    model.to(device)
    X_train = Variable(torch.from_numpy(X_train).float()).to(device)
    y_train = Variable(torch.from_numpy(y_train).float()).to(device)
    loss_list = []

    for epoch in range(epochs):
        # 初始化隐藏状态和细胞状态
        batch_size = X_train.size(0)
        h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
        c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

        outputs = model(X_train, (h0, c0))
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

def predict_lstm_model(model, model_name, X_test, y_test):
    model.load_state_dict(torch.load(model_name))  # 读取模型
    model.eval()
    model.to(device)
    print('test data: ', len(X_test))

    X_test = Variable(torch.from_numpy(X_test).float()).to(device)
    # 初始化隐藏状态和细胞状态
    batch_size = X_test.size(0)
    h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
    c0 = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

    pred = model(X_test, (h0, c0)).detach().cpu().numpy()

    # 评估
    print('MSE:', mean_squared_error(y_test, pred))
    print('MAE:', mean_absolute_error(y_test, pred))
    print('R2:', r2_score(y_test, pred))

# 设置超参数
seq_length = 10
input_size = 6
hidden_size = 64
num_layers = 3
output_size = 6
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义用于训练和测试的数据
trainnumber = 1
testnumber = 1
trainingFile = 'H:\\data\\softSensor\\merged{}.csv'.format(trainnumber)
testFile = 'H:\\data\\softSensor\\merged{}.csv'.format(testnumber)

# 加载数据
X_train, X_test, y_train, y_test = loadData(True, 0.2, trainingFile, trainingFile, seq_length)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型、优化器和损失函数
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    # 训练阶段
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 初始化隐藏状态和细胞状态 (确保维度正确)
            batch_size = inputs.size(0)
            h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

            # 前向传播
            outputs = model(inputs, (h0, c0))  # 将 h0 和 c0 传递给模型
            loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # 测试阶段
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 初始化隐藏状态和细胞状态
            batch_size = inputs.size(0)
            h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

            outputs = model(inputs, (h0, c0))  # 传递 hidden 参数
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()

    epoch_test_loss = running_test_loss / len(test_loader)
    test_losses.append(epoch_test_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}')

# 绘制损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 评估模型
model.eval()
with torch.no_grad():
    # 初始化隐藏状态和细胞状态
    batch_size = X_test.size(0)
    h0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    c0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    y_pred = model(X_test.to(device), (h0, c0))  # 传递 hidden 参数
    mse = mean_squared_error(y_test.cpu(), y_pred.cpu())
    print(f'Mean Squared Error: {mse:.4f}')