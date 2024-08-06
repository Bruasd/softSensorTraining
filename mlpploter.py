import torch
import torch.nn as nn
from torchviz import make_dot

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型实例
model = MLP()

# 使用随机数据生成计算图
x = torch.randn(1, 784)
y = model(x)

# 使用 make_dot 生成计算图，并保存为 PDF 文件
graph = make_dot(y, params=dict(model.named_parameters()))
graph.render("mlp_model", format="pdf")

# 使用 Netron 可视化计算图
# 在命令行中运行 `netron` 命令，然后打开 `mlp_model.pdf` 文件