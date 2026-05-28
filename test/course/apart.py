import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================= 1. 数据准备 =================
df = pd.read_csv('test_data\\student\\student-mat.csv', sep=';')
# 依然使用这几个最强特征
features = ['age', 'absences', 'G1', 'G2']
X_raw = df[features].values
# 【变化1】：将 G3 分数转化为 0 和 1 的标签
y_raw = (df['G3'] >= 10).astype(int).values
# 数据标准化 (分类任务同样非常需要)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
# 转为 Tensor。注意标签 Y 的形状要变成 (样本数, 1) 以匹配模型输出
X = torch.tensor(X_scaled, dtype=torch.float32)
Y = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1) 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# ================= 2. 定义分类模型 =================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 【变化2】：加入 Sigmoid 压缩机
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)      # 把线性结果变成 0-1 的概率
        return out
model = LogisticRegressionModel(X.shape[1])
# ================= 3. 损失函数与优化器 =================
# 【变化3】：不再用 MSELoss，改用 BCELoss (二元交叉熵)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 分类任务学习率可以稍微给大一点

# ================= 4. 开始训练 =================
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ================= 5. 模型评估 (看准确率) =================
model.eval()
with torch.no_grad():
    # 模型输出的是概率，比如 0.82
    test_probabilities = model(X_test)
    
    # 设定阈值：概率 >= 0.5 我们就认为预测它是及格 (1)
    predicted_classes = (test_probabilities >= 0.5).float()
    
    # 【变化4】：计算准确率 (预测对的个数 / 总个数)
    correct_predictions = (predicted_classes == y_test).sum()
    total_samples = y_test.size(0)
    accuracy = correct_predictions / total_samples
    
    print(f'\n--- 最终评估 ---')
    print(f'测试集准确率 (Accuracy): {accuracy.item() * 100:.2f}%')