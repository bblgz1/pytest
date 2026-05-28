import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 1. 读取空气质量数据 (注意分隔符和缺失值)
df = pd.read_csv('test_data\\air+quality\\AirQualityUCI.csv', sep=';', decimal=',')
# 删除最后两个全是空值的列（原文件末尾有多余的 ';' 和空字段）
df = df.iloc[:, :-2]
# 2. 缺失值处理：-200 表示缺失或仪器未检测到，替换为 NaN 并删除
df = df.replace(-200, np.nan).dropna()
# 3. 特征选择与目标定义
feature_cols = ['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)',
                'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
target_col = 'CO(GT)'
X_raw = df[feature_cols].astype(float)
y_raw = df[target_col].astype(float)
# 4. 数据标准化 (所有特征都是数值型，无需 LabelEncoder)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
# 转换为 PyTorch 张量
X = torch.tensor(X_scaled, dtype=torch.float32)
Y = torch.tensor(y_raw.values, dtype=torch.float32).reshape(-1, 1)  # 保持列向量

print(f"特征矩阵形状: {X.shape}")   # 例如 (9357, 8)
print(f"标签形状: {Y.shape}")
# 5. （可选）划分训练集和测试集
torch.manual_seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)# 训练集占 80%，测试集占 20%
# 6. 定义线性回归模型）
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)# 定义线性层，输入维度是特征数，输出维度是1

    def forward(self, x):
        return self.linear(x)

# 输入维度 = 特征数 = 8
model = LinearRegressionModel(input_dim=X.shape[1])# 定义模型
criterion = nn.MSELoss()# 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 7. 训练模型（使用训练集）
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()# 进入训练模式
    predictions = model(X_train).squeeze()# 前向传播,得到预测值
    loss = criterion(predictions, Y_train.squeeze())# 计算损失
    optimizer.zero_grad()# 梯度清零
    loss.backward()# 反向传播计算梯度
    optimizer.step()# 更新参数
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 8. 在测试集上评估
model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze()# 前向传播
    test_loss = criterion(test_pred, Y_test.squeeze())# 计算损失
    print(f'\n测试集均方误差 (MSE): {test_loss.item():.4f}')
    print(f'学习到的权重: {model.linear.weight.data.numpy().flatten()}')
    print(f'学习到的偏置: {model.linear.bias.data.item():.4f}')