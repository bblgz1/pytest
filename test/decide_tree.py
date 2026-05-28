# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 设置中文字体（防止乱码，以 Windows 为例，也可根据系统调整）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


iris = load_iris()
# 特征名翻译
feature_names_cn = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
X = pd.DataFrame(iris.data, columns=feature_names_cn)   
y = pd.Series(iris.target, name='target')

print("特征矩阵形状:", X.shape)      # (150, 4)
print("标签分布:\n", y.value_counts())

# 类别名翻译
target_names_cn = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# ------------------------------
# 3. 创建并训练决策树模型
# ------------------------------
clf = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# ------------------------------
# 4. 模型评估
# ------------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names_cn))
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ------------------------------
# 5. 可视化：决策树结构（中文）
# ------------------------------
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names_cn, class_names=target_names_cn,
          filled=True, rounded=True, fontsize=10)
plt.title("鸢尾花数据集决策树 (最大深度=3)", fontsize=14)
plt.tight_layout()
plt.savefig('iris_decision_tree_cn.png', dpi=150)
plt.show()

# ------------------------------
# 6. 可视化：特征重要性（中文条形图）
# ------------------------------
importance = clf.feature_importances_
features = feature_names_cn

plt.figure(figsize=(8, 5))
plt.barh(features, importance, color='steelblue')
plt.xlabel('重要性')
plt.title('决策树特征重要性')
for i, v in enumerate(importance):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center')
plt.tight_layout()
plt.savefig('iris_feature_importance_cn.png', dpi=150)
plt.show()

# ------------------------------
# 7. 可视化：混淆矩阵热力图（中文标签）
# ------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_cn, yticklabels=target_names_cn)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.tight_layout()
plt.savefig('iris_confusion_matrix_cn.png', dpi=150)
plt.show()