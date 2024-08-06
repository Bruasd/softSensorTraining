import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def preprocess_data(file_path, sequence_length):
    data = pd.read_csv(file_path)
    X, y = [], []
    unique_indices = data['Index'].unique()

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_indices)

    for index in unique_indices:
        subset = data[data['Index'] == index]
        features = subset[['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']]
        target = subset['Index'].iloc[0]

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        for i in range(len(features_scaled) - sequence_length):
            X.append(features_scaled[i:i + sequence_length])
            y.append(target)

    X = np.array(X)
    y = label_encoder.transform(y)

    return X, y, label_encoder

def preprocess_multiple_data(file_paths, sequence_length):
    X, y = [], []
    label_encoder = LabelEncoder()
    all_indices = []

    # 收集所有文件中的唯一索引
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        unique_indices = data['Index'].unique()
        all_indices.extend(unique_indices)

    # 为所有索引拟合 LabelEncoder
    label_encoder.fit(all_indices)

    # 预处理每个文件的数据
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        unique_indices = data['Index'].unique()

        for index in unique_indices:
            subset = data[data['Index'] == index]
            features = subset[['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']]
            target = subset['Index'].iloc[0]

            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)

            for i in range(len(features_scaled) - sequence_length):
                X.append(features_scaled[i:i + sequence_length])
                y.append(target)

    X = np.array(X)
    y = label_encoder.transform(y)

    return X, y, label_encoder


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        lstm_out = self.dropout(lstm_out[:, -1, :])
        predictions = self.linear(lstm_out)
        return predictions

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_accuracy = evaluate_model(model, val_loader, device)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            _, predicted = torch.max(y_pred.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # 保存预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # 打印每个类别的准确率
    for i, accuracy in enumerate(class_accuracies):
        print(f"类别 {i+1} 的准确率: {accuracy:.4f}")

    # 返回平均准确率
    return correct / total


if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('No GPU available, using CPU instead.')

# 使用示例
dic = 'H:\\data\\brushFTarea\\softelectric\\'
sequence_length = 10
batch_size = 32
epochs = 30
learning_rate = 0.001

file_paths=[]
j=6
for k in range(j):
    file_paths.append(dic+'ft{}_interpolated.csv'.format(k+1))

# X_train, y_train, label_encoder = preprocess_multiple_data(file_paths, sequence_length)
X_train, y_train, label_encoder = preprocess_multiple_data(file_paths[:-1], sequence_length)
X_val, y_val, label_encoder1 = preprocess_data(file_paths[-1], sequence_length)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = 6  # Number of features
hidden_layer_size = 30
output_size = len(label_encoder.classes_)  # Number of unique classes
num_layers = 2  # LSTM 层数

model = LSTMModel(input_size, hidden_layer_size, output_size, num_layers).to(device)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')

