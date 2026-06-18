import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 1. 加载 & 标准化
# ------------------------------
iris = load_iris()
X, y = iris.data, iris.target
feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
target_names  = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']

# KNN 基于距离，必须标准化（否则量纲大的特征主导距离计算）标准差标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")

# ------------------------------
# 2. 不同 k 值的准确率对比
# ------------------------------
k_values = [1, 3, 5, 7, 9, 11, 15, 20, 25]
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))
    print(f"k = {k:2d}  |  训练准确率: {train_scores[-1]:.3f}  |  测试准确率: {test_scores[-1]:.3f}")

# ------------------------------
# 3. 可视化：准确率 vs k
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_values, train_scores, 'o-', label='训练准确率', color='steelblue')
plt.plot(k_values, test_scores, 's-', label='测试准确率', color='darkorange')
plt.axvline(x=k_values[np.argmax(test_scores)],
            color='green', linestyle='--', alpha=0.7,
            label=f'最佳 k = {k_values[np.argmax(test_scores)]}')
plt.xlabel('k (邻居数)')
plt.ylabel('准确率')
plt.title('KNN: 不同 k 值对准确率的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/knn_k_comparison.png', dpi=150)
print("k 值对比图已保存: outputs/knn_k_comparison.png")

# ------------------------------
# 4. 交叉验证：选最佳 k
# ------------------------------
best_k = k_values[np.argmax(test_scores)]
print(f"\n最优 k (测试集): {best_k}")

cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    cv_scores.append(scores.mean())

best_k_cv = k_values[np.argmax(cv_scores)]
print(f"最优 k (5-折交叉验证): {best_k_cv}")

# ------------------------------
# 5. 最佳模型评估
# ------------------------------
best_knn = KNeighborsClassifier(n_neighbors=best_k_cv)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

print(f"\n最佳 KNN 模型 (k={best_k_cv}) 测试准确率: {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('预测')
plt.ylabel('真实')
plt.title(f'KNN (k={best_k_cv}) 混淆矩阵')
plt.tight_layout()
plt.savefig('outputs/knn_confusion_matrix.png', dpi=150)
print("混淆矩阵已保存: outputs/knn_confusion_matrix.png")

# ------------------------------
# 6. 决策边界可视化（取前两个特征）
# ------------------------------
X_2d = X_scaled[:, :2]  # 只用花萼长度和宽度
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, k in enumerate([1, 3, 5, 7, 11, 15]):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_2d, y)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = axes[i]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                         cmap='viridis', edgecolor='k', s=50)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f'k = {k}')

plt.suptitle('KNN 决策边界随 k 值的变化（前两个特征）', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/knn_decision_boundary.png', dpi=150)
print("决策边界图已保存: outputs/knn_decision_boundary.png")

# ------------------------------
# 7. 到 k 个邻居的距离可视化
# ------------------------------
sample_idx = 0  # 取第一个测试样本
sample = X_test[sample_idx].reshape(1, -1)
distances, indices = best_knn.kneighbors(sample, n_neighbors=best_k_cv)

plt.figure(figsize=(8, 5))
neighbor_labels = y_train[indices[0]]
colors = [f'C{i}' for i in neighbor_labels]
bars = plt.bar(range(1, best_k_cv + 1), distances[0], color=colors)
plt.xlabel('邻居排名')
plt.ylabel('欧氏距离')
plt.title(f'测试样本 #{sample_idx} 到 {best_k_cv} 个最近邻居的距离')
plt.xticks(range(1, best_k_cv + 1))
# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=f'C{i}', label=target_names[i]) for i in range(3)]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('outputs/knn_neighbor_distances.png', dpi=150)
print("邻居距离图已保存: outputs/knn_neighbor_distances.png")

print("\n=== KNN 实验完成 ===")
