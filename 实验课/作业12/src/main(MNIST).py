import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
from collections import OrderedDict

# ========== 任务2：加载图像数据集 ========== 
print("=" * 60)
print("任务 2：加载图像数据集")
print("=" * 60)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载完整训练集
train_full = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# 加载测试集
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 划分训练集和验证集 (80% 训练, 20% 验证)
train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("✓ 数据加载完成！\n")

# ========== 任务3：定义 CNN 模型 ==========
print("=" * 60)
print("任务 3：定义 CNN 模型")
print("=" * 60)

class MNIST_CNN(nn.Module):
    """
    针对 MNIST 手写数字识别的 CNN 模型
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNIST_CNN()
print(model)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print("✓ 模型定义完成！\n")

# ========== 任务4：训练模型、任务5：验证模型 ==========
print("=" * 60)
print("任务 4：训练模型")
print("=" * 60)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备: {device}")

# 4.1 选择损失函数
criterion = nn.CrossEntropyLoss()
print(f"损失函数: CrossEntropyLoss")

# 4.2 选择优化器
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"优化器: Adam (学习率 = {learning_rate})")

# 4.3 训练参数
num_epochs = 5  # 至少训练 5 个 epoch
print(f"训练轮数: {num_epochs} epochs")
print(f"批次大小: {batch_size}")

# 记录训练历史
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("\n开始训练...")

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 将数据移动到设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算训练集的平均损失和准确率
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    # 打印每个 epoch 的结果
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  训练 - Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%")
    print(f"  验证 - Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.2f}%")
    print()

print("训练完成！")

# ========== 训练结果可视化 ==========

# 1. 绘制损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss', marker='o')
plt.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# 2. 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, 'b-', label='Training Accuracy', marker='o')
plt.plot(range(1, num_epochs+1), val_accuracies, 'r-', label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

print("✓ 训练曲线已保存为 'training_history.png'")

# ========== 任务6：测试模型 ==========
print("=" * 60)
print("任务 6：测试模型")
print("=" * 60)

# 设置模型为评估模式
model.eval()

# 测试集评估
test_loss = 0.0
test_correct = 0
test_total = 0

# 用于存储预测结果和真实标签
all_predictions = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 统计损失和准确率
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # 保存结果用于可视化
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_images.extend(images.cpu())

# 计算测试集平均损失和准确率
avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * test_correct / test_total

print(f"\n测试集评估结果:")
print(f"  测试集 Loss: {avg_test_loss:.4f}")
print(f"  测试集 Accuracy: {test_accuracy:.2f}%")
print(f"  正确预测: {test_correct}/{test_total}")

# ========== 显示测试图像及预测结果 ==========

# 显示至少 8 张测试图像
num_images = 8
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.ravel()

for i in range(num_images):
    # 获取图像、真实标签和预测标签
    image = all_images[i]
    true_label = all_labels[i]
    pred_label = all_predictions[i]
    
    # 反归一化显示（MNIST: mean=0.1307, std=0.3081）
    img_display = image * 0.3081 + 0.1307
    img_display = img_display.squeeze().numpy()  # 去除通道维度
    
    # 显示图像
    axes[i].imshow(img_display, cmap='gray')
    
    # 设置标题，正确预测用绿色，错误预测用红色
    if true_label == pred_label:
        color = 'green'
        title = f'True: {true_label} | Pred: {pred_label} ✓'
    else:
        color = 'red'
        title = f'True: {true_label} | Pred: {pred_label} ✗'
    
    axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('MNIST Test Set Predictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ 测试集预测图像已保存为 'test_predictions.png'")

# ========== 最终总结 ========== 
print("\n" + "=" * 60)
print("任务完成总结")
print("=" * 60)
print(f"✓ 任务2: 数据加载完成 ({len(train_dataset)} 训练, {len(val_dataset)} 验证, {len(test_dataset)} 测试)")
print(f"✓ 任务3: CNN 模型定义完成 ({sum(p.numel() for p in model.parameters()):,} 参数)")
print(f"✓ 任务4: 模型训练完成 ({num_epochs} epochs)")
print(f"✓ 任务5: 模型验证完成 (最佳验证准确率: {max(val_accuracies):.2f}%)")
print(f"✓ 任务6: 模型测试完成 (测试准确率: {test_accuracy:.2f}%)")
print("=" * 60)
print("所有任务完成！")
print("=" * 60)



"""
进阶任务 1：修改网络结构
比较不同网络结构对 MNIST 分类性能的影响
"""

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ========== 数据加载（与之前相同） ==========
print("=" * 60)
print("进阶任务 1：修改网络结构")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"数据加载完成: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}\n")

# ========== 定义不同的模型结构 ==========

# 1. 原始模型（Baseline）
class BaselineCNN(nn.Module):
    """原始模型：2层卷积 + 2层全连接"""
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. 增加卷积层数量（3层卷积）
class DeeperCNN(nn.Module):
    """更深模型：3层卷积 + 2层全连接"""
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 新增
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 经过3次池化: 28->14->7->3
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # 第三次池化
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. 增加卷积核数量（更宽）
class WiderCNN(nn.Module):
    """更宽模型：增加卷积核数量"""
    def __init__(self):
        super(WiderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)   # 32 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64 -> 128
        self.fc1 = nn.Linear(128 * 7 * 7, 256)                    # 128 -> 256
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. 修改全连接层（更大）
class LargerFCCNN(nn.Module):
    """更大全连接层：增加神经元数量"""
    def __init__(self):
        super(LargerFCCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # 128 -> 512
        self.fc2 = nn.Linear(512, 256)          # 新增一层
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)          # 增加 dropout 比率
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 5. 增加 Dropout（更强正则化）
class StrongDropoutCNN(nn.Module):
    """强 Dropout 模型：更高的 dropout 比率"""
    def __init__(self):
        super(StrongDropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.5)   # 增加 dropout
        self.dropout2 = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

# ========== 训练函数 ==========
def train_model(model, model_name, num_epochs=5):
    """训练并评估模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用设备: {device}")
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    training_time = time.time() - start_time
    
    print(f"\n最终结果:")
    print(f"  最佳验证准确率: {max(val_accuracies):.2f}%")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"  训练时间: {training_time:.2f}秒")
    
    return {
        'name': model_name,
        'params': sum(p.numel() for p in model.parameters()),
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'best_val_acc': max(val_accuracies),
        'test_acc': test_acc,
        'training_time': training_time
    }

# ========== 定义所有模型 ==========
models = [
    (BaselineCNN(), "1. Baseline (原始模型)"),
    (DeeperCNN(), "2. Deeper (3层卷积)"),
    (WiderCNN(), "3. Wider (更多卷积核)"),
    (LargerFCCNN(), "4. Larger FC (更大全连接层)"),
    (StrongDropoutCNN(), "5. Strong Dropout (更强正则化)")
]

# ========== 训练所有模型 ==========
results = []

for model, name in models:
    result = train_model(model, name, num_epochs=5)
    results.append(result)

# ========== 结果比较 ==========
print("\n" + "=" * 80)
print("模型性能比较")
print("=" * 80)

print(f"\n{'模型名称':<30} {'参数量':<12} {'最佳验证准确率':<18} {'测试准确率':<12} {'训练时间':<10}")
print("-" * 80)

for result in results:
    print(f"{result['name']:<30} {result['params']:>10,}  {result['best_val_acc']:>16.2f}%  {result['test_acc']:>10.2f}%  {result['training_time']:>8.2f}s")

print("\n" + "=" * 80)
print("进阶任务 1 完成！")
print("=" * 80)


"""
进阶任务 2：比较不同优化器
比较 SGD 和 Adam 优化器在不同学习率下的性能
"""

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("进阶任务 2：比较不同优化器（SGD vs Adam）")
print("=" * 80)

# ========== 数据加载 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n数据加载完成:")
print(f"  训练集: {len(train_dataset)} 张")
print(f"  验证集: {len(val_dataset)} 张")
print(f"  测试集: {len(test_dataset)} 张")

# ========== 定义 CNN 模型 ==========
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========== 训练函数 ==========
def train_model(model, optimizer_name, learning_rate, num_epochs=5):
    """使用指定优化器训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 选择优化器
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        if epoch == num_epochs - 1:  # 只打印最后一个 epoch
            print(f"  Epoch {epoch+1}: 训练准确率={train_acc:.2f}%, 验证准确率={val_acc:.2f}%")
    
    # 测试阶段
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    training_time = time.time() - start_time
    
    return {
        'optimizer': optimizer_name,
        'lr': learning_rate,
        'val_accuracies': val_accuracies,
        'best_val_acc': max(val_accuracies),
        'test_acc': test_acc,
        'training_time': training_time
    }

# ========== 实验1：SGD 不同学习率比较 ==========
print("\n" + "=" * 80)
print("实验 1：SGD 优化器 - 不同学习率比较 (momentum=0.9)")
print("=" * 80)

sgd_lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
results_sgd = []

for lr in sgd_lrs:
    print(f"\n训练 SGD (学习率={lr})")
    model = MNIST_CNN()
    result = train_model(model, 'SGD', lr, num_epochs=5)
    results_sgd.append(result)
    print(f"  最佳验证准确率: {result['best_val_acc']:.2f}%")
    print(f"  测试准确率: {result['test_acc']:.2f}%")
    print(f"  训练时间: {result['training_time']:.2f}秒")

# ========== 实验2：Adam 不同学习率比较 ==========
print("\n" + "=" * 80)
print("实验 2：Adam 优化器 - 不同学习率比较")
print("=" * 80)

adam_lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001]
results_adam = []

for lr in adam_lrs:
    print(f"\n训练 Adam (学习率={lr})")
    model = MNIST_CNN()
    result = train_model(model, 'Adam', lr, num_epochs=5)
    results_adam.append(result)
    print(f"  最佳验证准确率: {result['best_val_acc']:.2f}%")
    print(f"  测试准确率: {result['test_acc']:.2f}%")
    print(f"  训练时间: {result['training_time']:.2f}秒")

# ========== 打印结果表格 ==========

print("\n" + "=" * 80)
print("实验结果汇总")
print("=" * 80)

print("\n表1：SGD 优化器不同学习率结果")
print("-" * 60)
print(f"{'学习率':<12} {'最佳验证准确率':<20} {'测试准确率':<15} {'训练时间(秒)':<12}")
print("-" * 60)
for result in results_sgd:
    print(f"lr={result['lr']:<8} {result['best_val_acc']:>17.2f}%  {result['test_acc']:>12.2f}%  {result['training_time']:>10.2f}")

print("\n表2：Adam 优化器不同学习率结果")
print("-" * 60)
print(f"{'学习率':<12} {'最佳验证准确率':<20} {'测试准确率':<15} {'训练时间(秒)':<12}")
print("-" * 60)
for result in results_adam:
    print(f"lr={result['lr']:<8} {result['best_val_acc']:>17.2f}%  {result['test_acc']:>12.2f}%  {result['training_time']:>10.2f}")

# 找出最佳配置
best_sgd = max(results_sgd, key=lambda x: x['test_acc'])
best_adam = max(results_adam, key=lambda x: x['test_acc'])

print("\n" + "=" * 80)
print("最佳配置总结")
print("=" * 80)
print(f"\nSGD 最佳学习率: {best_sgd['lr']}")
print(f"  - 最佳验证准确率: {best_sgd['best_val_acc']:.2f}%")
print(f"  - 测试准确率: {best_sgd['test_acc']:.2f}%")

print(f"\nAdam 最佳学习率: {best_adam['lr']}")
print(f"  - 最佳验证准确率: {best_adam['best_val_acc']:.2f}%")
print(f"  - 测试准确率: {best_adam['test_acc']:.2f}%")

print("\n" + "=" * 80)
print("进阶任务 2 完成！")
print("=" * 80)
