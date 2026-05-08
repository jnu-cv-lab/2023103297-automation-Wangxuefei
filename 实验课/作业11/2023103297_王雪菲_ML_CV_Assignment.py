from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ========== 任务1：数据准备 ========== 
# 加载数据集
digits = load_digits()

# 查看数据集中图像的数量
print(f"图像总数: {len(digits.images)}")

# 查看每张图像的大小
print(f"每张图像的大小: {digits.images[0].shape}")

# 查看类别标签
print(f"所有类别标签: {np.unique(digits.target)}")

# 显示若干张样本图像及其真实标签
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# ========== 任务2：数据划分 ==========
from sklearn.model_selection import train_test_split

# 特征数据（图像）和标签
X = digits.data   # 形状 (1797, 64)
y = digits.target

# 划分训练集和测试集，测试集比例 25%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ========== 任务3：特征表示 ==========
# 查看一张图像如何变成向量
print(f"原始图像形状: {digits.images[0].shape}")
print(f"特征向量形状: {digits.data[0].shape}")
print(f"特征向量内容: {digits.data[0]}")

# ========== 任务4：模型训练 ==========
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', gamma='scale', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} 测试准确率:{acc:.4f}")

# ========== 任务5：结果比较 ==========
print("\n模型测试准确率汇总：")
print("-" * 35)
print(f"{'模型':<20} {'准确率':<10}")
print("-" * 35)
for name, acc in accuracies.items():
    print(f"{name:<20} {acc:.4f}")
print("-" * 35)

best_model = max(accuracies, key=accuracies.get)
print(f"\n最佳模型: {best_model} (准确率: {accuracies[best_model]:.4f})")

# ========== 任务6：错误样本分析 ==========
# 以最佳模型SVM为例，分析错误样本
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵形状: {cm.shape}")

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('SVM Confusion Matrix - Handwritten Digit Classification', fontsize=14)
plt.colorbar()

# 设置坐标轴
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10), fontsize=10)
plt.yticks(tick_marks, range(10), fontsize=10)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

# 在格子中添加数字
thresh = cm.max() / 2
for i in range(10):
    for j in range(10):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# 打印混淆矩阵数值
print("\n混淆矩阵（数值形式）：")
print("真实↓  预测→", " ".join([f"{i:3d}" for i in range(10)]))
print("-" * 60)
for i in range(10):
    print(f"       {i}     " + " ".join([f"{cm[i,j]:3d}" for j in range(10)]))

# 错误样本分析
# 找出所有错误分类的索引
error_indices = np.where(y_pred != y_test)[0]
print(f"总错误样本数: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

# 按混淆类型分组收集错误样本
errors_by_pair = {}
for idx in error_indices:
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    pair = (true_label, pred_label)
    if pair not in errors_by_pair:
        errors_by_pair[pair] = []
    errors_by_pair[pair].append(idx)

# 展示最常见的几种错误类型（每种最多3个样本）
print("\n错误类型分析：")
for pair, indices in sorted(errors_by_pair.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
    print(f"  真实值 {pair[0]} → 预测值 {pair[1]}: {len(indices)} 个样本")

# 可视化错误样本
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for i, (pair, indices) in enumerate(sorted(errors_by_pair.items(), key=lambda x: len(x[1]), reverse=True)[:15]):
    if i < len(axes):
        idx = indices[0]  # 取第一个错误样本
        ax = axes[i]
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        ax.set_title(f"True:{pair[0]} Pred:{pair[1]}", fontsize=10, color='red')
        ax.axis('off')

# 隐藏多余的子图
for j in range(len(errors_by_pair), len(axes)):
    axes[j].axis('off')

plt.suptitle('SVM Misclassified Samples Examples', fontsize=14)
plt.tight_layout()
plt.show()

# 找出最容易混淆的数字对
# 找出非对角线上最大的错误分类数量
max_error = 0
most_confused_pair = None
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > max_error:
            max_error = cm[i, j]
            most_confused_pair = (i, j)

print(f"最容易混淆的数字对: {most_confused_pair[0]} 被误判为 {most_confused_pair[1]}")
print(f"错误数量: {max_error} 次")

# 计算每个数字的召回率和精确率
print("\n各类别分类表现：")
print("-" * 60)
print(f"{'数字':<6} {'召回率(Recall)':<15} {'精确率(Precision)':<15} {'错误数':<10}")
print("-" * 60)

for i in range(10):
    # 召回率 = 正确识别的该数字数量 / 该数字实际总数
    recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    # 精确率 = 正确识别的该数字数量 / 预测为该数字的总数
    precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    errors = cm[i].sum() - cm[i, i]
    print(f"{i:<6}   {recall:<15.4f}     {precision:<15.4f}    {errors:<10}")
