import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from datapro import train,test, n_steps

from datapro import create_combined_train_loader

from sklearn.metrics import f1_score, classification_report

# class MLP(nn.Module):
#     def __init__(self, input_features):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_features, 64)  # Adjusted for correct number of input features
#         self.dropout1 = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(64, 64)
#         self.dropout2 = nn.Dropout(0.1)
#         self.fc3 = nn.Linear(64, 8)  # Adjust if different number of classes needed
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = F.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x

class MLP(nn.Module):
    def __init__(self, input_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)  # Adjusted for correct number of input features
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 64)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(64, 8)  # Adjust if different number of classes needed

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set the number of input features based on your sliding window data configuration
input_features = 6 # This should match n_steps * number of features per step
model = MLP(input_features)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(device, model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model.to(device)
    model.train()

    train_losses = []  # 用于存储训练集上的损失值
    test_losses = []   # 用于存储测试集上的损失值

    best_accuracy = 0.0  # 初始化最佳准确率
    best_model_state_dict = None  # 初始化最佳模型参数

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        accuracy,all_labels, all_preds= evaluate_model(device, model, test_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%')

        # 保存准确率最高的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_labels=all_labels
            best_preds=all_preds
            best_model_state_dict = model.state_dict().copy()

    # 加载最佳模型
    model.load_state_dict(best_model_state_dict)

    # 绘制混淆矩阵
    plot_confusion_matrix(best_labels, best_preds)


# Evaluation function
def evaluate_model(device, model, test_loader):
    model.eval()
    model.to(device)  # 将模型移动到指定的设备
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # 将输入数据移动到设备
            labels = labels.to(device)  # 将标签数据移动到设备
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 保存预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 计算每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # 打印每个类别的准确率
    for i, accuracy in enumerate(class_accuracies):
        print(f"类别 {i+1} 的准确率: {accuracy:.4f}")

    # 返回平均准确率
    accuracy = 100 * correct / total
    return accuracy , all_preds, all_labels

# 绘制混淆矩阵函数
def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    # 将混淆矩阵转换为百分比
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="YlGnBu")

    # 修改 x 轴和 y 轴的刻度标签
    ax.set_xticklabels(np.arange(1, 9))
    ax.set_yticklabels(np.arange(1, 9))

    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

# dic = 'H:\\data\\brushFTarea\\softelectric\\'#这是直接使用传感器测刷牙区域分类
dic = "H:\\data\\brushSoftSensor\\"#这是在传感器上放了个柔性传感器再测
file_paths=[]
j=4
for k in range(j):
    file_paths.append(dic+'ft{}_interpolated.csv'.format(k+1))
# train_comb = create_combined_train_loader(file_paths[:-1])
train_comb = create_combined_train_loader(file_paths)
train = train_comb

test = create_combined_train_loader([file_paths[-1]])
train_model(device, model, train, test, criterion, optimizer)



