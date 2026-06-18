import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 修复中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 1. 超参数
# ------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 15              # MNIST 比 FashionMNIST 简单，收敛更快
INPUT_DIM = 28 * 28          # 784
HIDDEN1_DIM = 256
HIDDEN2_DIM = 128
OUTPUT_DIM = 10              # 数字 0-9

# ------------------------------
# 2. 加载 MNIST 数据
# ------------------------------
# yann.lecun.com 服务器不稳定，换用 PyTorch 官方 AWS 镜像
from torchvision.datasets import MNIST
MNIST.mirrors = ['https://ossci-datasets.s3.amazonaws.com/mnist/']

transform = transforms.ToTensor()

train_data = MNIST(
    root='data', train=True, download=True, transform=transform)
test_data = MNIST(
    root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集: {len(train_data)} 张, 测试集: {len(test_data)} 张")

# ------------------------------
# 3. 定义 MLP 模型
# ------------------------------
class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1_DIM)
        self.fc2 = nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM)
        self.fc3 = nn.Linear(HIDDEN2_DIM, OUTPUT_DIM)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, INPUT_DIM)          # 展平: (B,1,28,28) -> (B,784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = DigitMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)

# ------------------------------
# 4. 训练 + 记录
# ------------------------------
train_losses, test_losses, test_accuracies = [], [], []

for epoch in range(NUM_EPOCHS):
    # ---- 训练阶段 ----
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # ---- 测试阶段 ----
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            running_test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_losses.append(running_test_loss / len(test_loader))
    accuracy = 100.0 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}]  "
          f"Train Loss: {train_losses[-1]:.4f}  "
          f"Test Loss: {test_losses[-1]:.4f}  "
          f"Accuracy: {accuracy:.1f}%")

# ------------------------------
# 5. 可视化：训练曲线
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(range(1, NUM_EPOCHS + 1), train_losses, label='训练损失', marker='o')
axes[0].plot(range(1, NUM_EPOCHS + 1), test_losses, label='测试损失', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('MNIST 训练 / 测试损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, NUM_EPOCHS + 1), test_accuracies, color='green', marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title(f'测试准确率 (最高: {max(test_accuracies):.2f}%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/mnist_training.png', dpi=150)
print("训练曲线已保存: outputs/mnist_training.png")

# ------------------------------
# 6. 可视化：预测样本
# ------------------------------
model.eval()
sample_images, sample_labels = next(iter(test_loader))
with torch.no_grad():
    sample_preds = torch.max(model(sample_images), 1)[1]

fig, axes = plt.subplots(4, 6, figsize=(12, 8))
axes = axes.flatten()

for i in range(24):
    img = sample_images[i].squeeze()
    true_label = sample_labels[i].item()
    pred_label = sample_preds[i].item()
    color = 'green' if pred_label == true_label else 'red'

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"真实: {true_label}  预测: {pred_label}",
                      fontsize=10, color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('outputs/mnist_predictions.png', dpi=150)
print("预测样本已保存: outputs/mnist_predictions.png")

# ------------------------------
# 7. 混淆矩阵
# ------------------------------
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        all_preds.extend(torch.max(model(images), 1)[1].tolist())
        all_labels.extend(labels.tolist())

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.xlabel('预测')
plt.ylabel('真实')
plt.title(f'MNIST 混淆矩阵 (准确率: {max(test_accuracies):.2f}%)')
plt.tight_layout()
plt.savefig('outputs/mnist_confusion.png', dpi=150)
print("混淆矩阵已保存: outputs/mnist_confusion.png")
