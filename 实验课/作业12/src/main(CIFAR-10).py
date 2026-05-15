"""
进阶任务 3：比较 MNIST 和 CIFAR-10
使用 CIFAR-10 完成图像分类任务，并与 MNIST 对比
"""

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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("进阶任务 3：CIFAR-10 图像分类实验")
print("=" * 80)

# ========== CIFAR-10 数据加载 ==========
print("\n" + "=" * 60)
print("1. 加载 CIFAR-10 数据集")
print("=" * 60)

# CIFAR-10 数据预处理（增强版）
# CIFAR-10 需要更复杂的数据增强来提高性能
transform_cifar_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载 CIFAR-10 数据集
train_full_cifar = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_cifar_train
)

test_dataset_cifar = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_cifar_test
)

# 划分训练集和验证集
train_size_cifar = int(0.8 * len(train_full_cifar))
val_size_cifar = len(train_full_cifar) - train_size_cifar
train_dataset_cifar, val_dataset_cifar = random_split(
    train_full_cifar, [train_size_cifar, val_size_cifar]
)

# CIFAR-10 类别名称
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"CIFAR-10 数据集统计:")
print(f"  训练集: {len(train_dataset_cifar)} 张")
print(f"  验证集: {len(val_dataset_cifar)} 张")
print(f"  测试集: {len(test_dataset_cifar)} 张")
print(f"  图像尺寸: 3x32x32 (RGB彩色图)")
print(f"  类别数量: 10")
print(f"  类别名称: {cifar_classes}")

# 创建 DataLoader
batch_size = 64
train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=batch_size, shuffle=True)
val_loader_cifar = DataLoader(val_dataset_cifar, batch_size=batch_size, shuffle=False)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=batch_size, shuffle=False)

print("\n✓ CIFAR-10 数据加载完成！")

# ========== 显示 CIFAR-10 样本图像 ==========
print("\n" + "=" * 60)
print("2. CIFAR-10 样本图像")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()

for i in range(8):
    image, label = train_dataset_cifar[i]
    # 反归一化显示
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img_display = image * std + mean
    img_display = torch.clamp(img_display, 0, 1)
    img_display = img_display.permute(1, 2, 0).numpy()
    
    axes[i].imshow(img_display)
    axes[i].set_title(f'Label: {cifar_classes[label]}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('CIFAR-10 Dataset Samples', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=150)
plt.show()
print("✓ CIFAR-10 样本图像已保存为 'cifar10_samples.png'")

# ========== 定义 CIFAR-10 CNN 模型 ==========
print("\n" + "=" * 60)
print("3. 定义 CIFAR-10 CNN 模型")
print("=" * 60)

class CIFAR10_CNN(nn.Module):
    """
    针对 CIFAR-10 的 CNN 模型（更深更宽）
    输入: 3x32x32，输出: 10个类别
    """
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        
        # 卷积块1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 卷积块2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 卷积块3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

model_cifar = CIFAR10_CNN()
print(model_cifar)
print(f"\n模型参数量: {sum(p.numel() for p in model_cifar.parameters()):,}")
print("✓ CIFAR-10 模型定义完成！")

# ========== 训练 CIFAR-10 模型 ==========
print("\n" + "=" * 60)
print("4. 训练 CIFAR-10 模型")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cifar = model_cifar.to(device)
print(f"使用设备: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cifar.parameters(), lr=0.001)

num_epochs = 10  # CIFAR-10 需要更多 epoch
print(f"训练轮数: {num_epochs} epochs")
print(f"批次大小: {batch_size}")

train_losses_cifar = []
train_accs_cifar = []
val_losses_cifar = []
val_accs_cifar = []

print("\n开始训练...")
print("-" * 60)

start_time = time.time()

for epoch in range(num_epochs):
    # 训练阶段
    model_cifar.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader_cifar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model_cifar(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_train_loss = running_loss / len(train_loader_cifar)
    epoch_train_acc = 100 * correct / total
    
    # 验证阶段
    model_cifar.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader_cifar:
            images, labels = images.to(device), labels.to(device)
            outputs = model_cifar(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    epoch_val_loss = val_loss / len(val_loader_cifar)
    epoch_val_acc = 100 * val_correct / val_total
    
    train_losses_cifar.append(epoch_train_loss)
    train_accs_cifar.append(epoch_train_acc)
    val_losses_cifar.append(epoch_val_loss)
    val_accs_cifar.append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  训练 - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%")
    print(f"  验证 - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

training_time_cifar = time.time() - start_time
print(f"\n训练完成！总耗时: {training_time_cifar:.2f}秒")

# ========== 测试 CIFAR-10 模型 ==========
print("\n" + "=" * 60)
print("5. 测试 CIFAR-10 模型")
print("=" * 60)

model_cifar.eval()
test_correct = 0
test_total = 0
test_loss = 0.0

all_predictions_cifar = []
all_labels_cifar = []
all_images_cifar = []

with torch.no_grad():
    for images, labels in test_loader_cifar:
        images, labels = images.to(device), labels.to(device)
        outputs = model_cifar(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        all_predictions_cifar.extend(predicted.cpu().numpy())
        all_labels_cifar.extend(labels.cpu().numpy())
        all_images_cifar.extend(images.cpu())

avg_test_loss_cifar = test_loss / len(test_loader_cifar)
test_accuracy_cifar = 100 * test_correct / test_total

print(f"\nCIFAR-10 测试结果:")
print(f"  测试集 Loss: {avg_test_loss_cifar:.4f}")
print(f"  测试集 Accuracy: {test_accuracy_cifar:.2f}%")
print(f"  正确预测: {test_correct}/{test_total}")

# ========== 显示 CIFAR-10 测试结果 ==========
num_images = 8
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.ravel()

for i in range(num_images):
    image = all_images_cifar[i]
    true_label = all_labels_cifar[i]
    pred_label = all_predictions_cifar[i]
    
    # 反归一化
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img_display = image * std + mean
    img_display = torch.clamp(img_display, 0, 1)
    img_display = img_display.permute(1, 2, 0).numpy()
    
    axes[i].imshow(img_display)
    
    if true_label == pred_label:
        color = 'green'
        title = f'True: {cifar_classes[true_label]}\nPred: {cifar_classes[pred_label]} ✓'
    else:
        color = 'red'
        title = f'True: {cifar_classes[true_label]}\nPred: {cifar_classes[pred_label]} ✗'
    
    axes[i].set_title(title, fontsize=9, color=color)
    axes[i].axis('off')

plt.suptitle('CIFAR-10 Test Set Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cifar10_test_predictions.png', dpi=150)
plt.show()

# ========== MNIST 结果（从之前的实验获取） ==========
mnist_test_accuracy = 98.80  # 替换为实际的 MNIST 测试准确率
mnist_params = 421642
cifar_params = sum(p.numel() for p in model_cifar.parameters())

# ========== 绘制训练曲线 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(range(1, num_epochs+1), train_losses_cifar, 'b-', label='Training Loss', marker='o')
axes[0].plot(range(1, num_epochs+1), val_losses_cifar, 'r-', label='Validation Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('CIFAR-10 Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# 准确率曲线
axes[1].plot(range(1, num_epochs+1), train_accs_cifar, 'b-', label='Training Accuracy', marker='o')
axes[1].plot(range(1, num_epochs+1), val_accs_cifar, 'r-', label='Validation Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('CIFAR-10 Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('cifar10_training_history.png', dpi=150)
plt.show()

# ========== 保存 CIFAR-10 模型 ==========
torch.save(model_cifar.state_dict(), 'cifar10_cnn_model.pth')
print("\n✓ CIFAR-10 模型已保存为 'cifar10_cnn_model.pth'")

# ========== MNIST vs CIFAR-10 对比分析 ==========
print("\n" + "=" * 80)
print("6. MNIST vs CIFAR-10 对比分析")
print("=" * 80)

print("\n数据集对比:")
print("-" * 80)
print(f"{'特征':<20} {'MNIST':<30} {'CIFAR-10':<30}")
print("-" * 80)
print(f"{'图像类型':<20} {'灰度图 (单通道)':<30} {'彩色图 (RGB三通道)':<30}")
print(f"{'图像尺寸':<20} {'28×28 像素':<30} {'32×32 像素':<30}")
print(f"{'类别数量':<20} {'10 (数字 0-9)':<30} {'10 (物体类别)':<30}")
print(f"{'训练集大小':<20} {'60,000 张':<30} {'50,000 张':<30}")
print(f"{'测试集大小':<20} {'10,000 张':<30} {'10,000 张':<30}")
print(f"{'模型参数量':<20} {f'{mnist_params:,}':<30} {f'{cifar_params:,}':<30}")
print(f"{'测试准确率':<20} {f'{mnist_test_accuracy:.2f}%':<30} {f'{test_accuracy_cifar:.2f}%':<30}")

print("\n" + "=" * 80)
print("进阶任务 3 完成！")
print("=" * 80)