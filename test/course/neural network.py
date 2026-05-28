import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 参数设置
n_in, n_h1, n_h2, n_out = 10, 20, 10, 1
n_samples = 10000                

# 生成输入数据 (10000, 10) 和标签 (10000, 1)
torch.manual_seed(42)            
x = torch.randn(n_samples, n_in)
# 标签规则：第一个特征 > 0 则标签为 1，否则为 0
y = (x[:, 0] > 0).float().unsqueeze(1)   # 变为列向量 (10000,1)

# 划分训练集和测试集 (70% 训练, 30% 测试)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

print(f"训练集大小: {x_train.shape[0]} 样本")
print(f"测试集大小: {x_test.shape[0]} 样本")

# 定义网络结构（两个隐藏层）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_in, n_h1)
        self.linear2 = nn.Linear(n_h1, n_h2)
        self.linear3 = nn.Linear(n_h2, n_out)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

net = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 1000

# 训练循环（使用训练集）
for epoch in range(num_epochs):
    net.train()
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}')

# 在测试集上评估
net.eval()
with torch.no_grad():
    test_outputs = net(x_test)
    test_loss = criterion(test_outputs, y_test)
    predicted = test_outputs.round()
    accuracy = (predicted == y_test).float().mean()
    print(f"\n测试集损失: {test_loss.item():.4f}")
    print(f"测试集准确率: {accuracy.item():.2%}")

    # 显示前10个测试样本的预测结果和真实标签
    print("\n前10个测试样本预测 vs 真实:")
    for i in range(10):
        print(f"预测: {predicted[i].item():.0f}, 真实: {y_test[i].item():.0f}")