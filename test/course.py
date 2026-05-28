#线性回归模型实验：
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. 读取数据 (注意这个数据集是用分号 ';' 分隔的)
df = pd.read_csv('test_data\\student\\student-mat.csv', sep=';')

# 2. 特征选择
# 为了简单起见，我们选几个代表性的特征：
# 数值型：age, absences, G1, G2
# 类别型：sex (F/M), internet (yes/no)
selected_features = ['age', 'absences', 'G1', 'G2', 'sex', 'internet']
X_raw = df[selected_features].copy()
y_raw = df['G3'].copy()

# 3. 处理类别型数据 (变成 0 和 1)
# sex: F -> 0, M -> 1 | internet: no -> 0, yes -> 1
le = LabelEncoder()
X_raw['sex'] = le.fit_transform(X_raw['sex'])
X_raw['internet'] = le.fit_transform(X_raw['internet'])

# 4. 数据标准化 (非常重要！否则 absences 的大数值会干扰 G1 这种小数值)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 5. 转换为 PyTorch 张量
X = torch.tensor(X_scaled, dtype=torch.float32)
Y = torch.tensor(y_raw.values, dtype=torch.float32)

# 打印一下形状，看看是不是和你之前的代码对上了
print(f"特征矩阵形状: {X.shape}") # 应该是 [395, 6]
print(f"标签形状: {Y.shape}")     # 应该是 [395]
torch.manual_seed(42)
X = torch.randn(100,2)#100个样本，每个样本两个特征
true_w = torch.tensor([2.0,3.0])
true_b = 4.0
Y = X @ true_w + true_b + 0.1*torch.randn(100)
print(X.shape,Y.shape)
print(X[:5],Y[:5])
w = torch.randn(2, requires_grad=True)
#定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入特征数为2，输出特征数为1

    def forward(self, x):#定义前向传播函数
        return self.linear(x)#定义损失函数和优化器
model = LinearRegressionModel()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
#训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    predictions = model(X).squeeze()  # 前向传播
    loss = criterion(predictions, Y)  # 计算损失
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Learned parameters: w={model.linear.weight.data}, b={model.linear.bias.data}')
#评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    predictions = model(X).squeeze()  # 前向传播
    mse = criterion(predictions, Y)  # 计算均方误差
    print(f'Mean Squared Error: {mse.item():.4f}')

