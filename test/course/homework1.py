import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 读取空气质量数据
df = pd.read_csv('test_data\\air+quality\\AirQualityUCI.csv', sep=';', decimal=',')
df = df.iloc[:, :-2]                     # 删除末尾两列空值
df = df.replace(-200, np.nan).dropna()   # 处理缺失值
# 2. 特征与目标
feature_cols = ['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S2(NMHC)',
                'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
target_col = 'CO(GT)'
X_raw = df[feature_cols].astype(float)
y_raw = df[target_col].astype(float)
# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X = torch.tensor(X_scaled, dtype=torch.float32)
Y = torch.tensor(y_raw.values, dtype=torch.float32).reshape(-1, 1)
# 4. 划分训练集和测试集
torch.manual_seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 5. 定义模型、损失函数、优化器
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 6. 训练并记录损失
num_epochs = 1000
train_losses = []          # 记录每个 epoch 的训练损失
for epoch in range(num_epochs):
    model.train()
    predictions = model(X_train).squeeze()
    loss = criterion(predictions, Y_train.squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# 7. 测试集评估
model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze().numpy()
    test_true = Y_test.squeeze().numpy()
    test_loss = np.mean((test_pred - test_true) ** 2)
    print(f'\n测试集均方误差 (MSE): {test_loss:.4f}')
    print(f'学习到的权重: {model.linear.weight.data.numpy().flatten()}')
    print(f'学习到的偏置: {model.linear.bias.data.item():.4f}')
# 8. 绘制损失曲线 (Loss Curve)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue', linewidth=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)
# 9. 绘制预测值 vs 真实值散点图 (Scatter Plot)
plt.subplot(1, 2, 2)
plt.scatter(test_true, test_pred, alpha=0.6, edgecolors='k', s=30)
# 添加 y=x 参考线
min_val = min(test_true.min(), test_pred.min())
max_val = max(test_true.max(), test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
plt.xlabel('True CO(GT)')
plt.ylabel('Predicted CO(GT)')
plt.title(f'Test Set Predictions vs True Values\nMSE = {test_loss:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/linear_regression_results.png', dpi=150)  # 保存图像
plt.show()