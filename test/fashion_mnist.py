import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 修复中文显示（Windows 自带黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

# ------------------------------
# 1. 超参数
# ------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
INPUT_DIM = 28 * 28  # 784
HIDDEN1_DIM = 256
HIDDEN2_DIM = 128
OUTPUT_DIM = 10       # 10 类服装

# ------------------------------
# 2. 加载 FashionMNIST 数据
# ------------------------------
transform = transforms.ToTensor()  # 自动归一化到 [0, 1]

train_data = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '包', '短靴']
print(f"训练集: {len(train_data)} 张, 测试集: {len(test_data)} 张")

# ------------------------------
# 3. 定义 MLP 模型
# ------------------------------
class FashionMLP(nn.Module):
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
        x = self.fc3(x)                    # 输出 logits，CrossEntropyLoss 内置 softmax
        return x

model = FashionMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)

# ------------------------------
# 4. 训练 + 记录
# ------------------------------
train_losses = []
test_losses = []
test_accuracies = []

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

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- 测试阶段 ----
    model.eval()
    running_test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_test_loss = running_test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    test_losses.append(avg_test_loss)
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}]  "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Test Loss: {avg_test_loss:.4f}  "
          f"Accuracy: {accuracy:.1f}%")

# ------------------------------
# 5. 可视化：训练曲线
# ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(range(1, NUM_EPOCHS + 1), train_losses, label='训练损失', marker='o')
axes[0].plot(range(1, NUM_EPOCHS + 1), test_losses, label='测试损失', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练 / 测试损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, NUM_EPOCHS + 1), test_accuracies, color='green', marker='o')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title(f'测试准确率 (最高: {max(test_accuracies):.1f}%)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fashion_mnist_training.png', dpi=150)
print("训练曲线已保存: outputs/fashion_mnist_training.png")

# ------------------------------
# 6. 可视化：预测样本展示
# ------------------------------
model.eval()
sample_images, sample_labels = next(iter(test_loader))
with torch.no_grad():
    sample_outputs = model(sample_images)
    _, sample_preds = torch.max(sample_outputs, 1)

fig, axes = plt.subplots(4, 6, figsize=(12, 8))
axes = axes.flatten()

for i in range(24):
    img = sample_images[i].squeeze()  # (28, 28)
    true_label = CLASS_NAMES[sample_labels[i]]
    pred_label = CLASS_NAMES[sample_preds[i]]
    color = 'green' if sample_preds[i] == sample_labels[i] else 'red'

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"真实: {true_label}\n预测: {pred_label}", fontsize=8, color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('outputs/fashion_mnist_predictions.png', dpi=150)
print("预测样本已保存: outputs/fashion_mnist_predictions.png")

# ------------------------------
# 7. 混淆矩阵
# ------------------------------
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title(f'混淆矩阵 (准确率: {max(test_accuracies):.1f}%)')
plt.tight_layout()
plt.savefig('outputs/fashion_mnist_confusion.png', dpi=150)
print("混淆矩阵已保存: outputs/fashion_mnist_confusion.png")
