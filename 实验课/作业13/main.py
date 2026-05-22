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

# ========== 任务1：复用上次 CNN 模型 ==========
print("=" * 80)
print("任务1：复用上次 CNN 模型 - CIFAR-10")
print("=" * 80)

# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# CIFAR-10 数据预处理
transform_cifar_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集
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

batch_size = 64
train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=batch_size, shuffle=True)
val_loader_cifar = DataLoader(val_dataset_cifar, batch_size=batch_size, shuffle=False)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=batch_size, shuffle=False)

print(f"数据集统计:")
print(f"  训练集: {len(train_dataset_cifar)} 张")
print(f"  验证集: {len(val_dataset_cifar)} 张")
print(f"  测试集: {len(test_dataset_cifar)} 张")


# 定义模型（与之前相同的架构）
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    训练模型并记录所有指标
    """
    model = model.to(device)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
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
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        
        # 验证阶段
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
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  训练 - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}%")
        print(f"  验证 - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")
    
    return train_losses, train_accs, val_losses, val_accs

# 测试函数
def test_model(model, test_loader, criterion, device='cpu'):
    """
    测试模型并返回测试准确率
    """
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_correct / test_total
    
    return avg_test_loss, test_accuracy

# ========== 实验配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")
num_epochs = 10
criterion = nn.CrossEntropyLoss()

# 重置随机种子
set_seed(42)
    
# 创建新模型
model = CIFAR10_CNN()

# 创建优化器（这里使用Adam作为示例）
optimizer = optim.Adam(model.parameters(), lr=0.001)
    
# 训练模型
start_time = time.time()
train_losses, train_accs, val_losses, val_accs = train_model(
    model, train_loader_cifar, val_loader_cifar, criterion, optimizer, num_epochs, device
)
training_time = time.time() - start_time

# 测试模型
test_loss, test_acc = test_model(model, test_loader_cifar, criterion, device)

print(f"\n最终结果:")
print(f"  训练准确率: {train_accs[-1]:.2f}%")
print(f"  验证准确率: {val_accs[-1]:.2f}%")
print(f"  测试准确率: {test_acc:.2f}%")
print(f"  训练时间: {training_time:.2f}秒")


# ========== 任务2：优化器对比实验 ==========
print("=" * 80)
print("任务2：优化器对比实验 - CIFAR-10")
print("=" * 80)

# 定义要对比的优化器
optimizers_config = {
    'SGD': {'optimizer': optim.SGD, 'lr': 0.01},
    'SGD+Momentum': {'optimizer': optim.SGD, 'lr': 0.01, 'momentum': 0.9},
    'Adam': {'optimizer': optim.Adam, 'lr': 0.001}
}

# 存储所有结果
results = {}

print("\n" + "=" * 60)
print("开始优化器对比实验")
print("=" * 60)

for opt_name in optimizers_config.keys():
    print(f"\n{'='*60}")
    print(f"使用优化器: {opt_name}")
    print(f"{'='*60}")
    
    # 重置随机种子确保公平比较
    set_seed(42)
    
    # 创建新模型
    model = CIFAR10_CNN()
    
    # 创建优化器
    config = optimizers_config[opt_name]
    if opt_name == 'SGD+Momentum':
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'], momentum=config['momentum'])
    else:
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])
    
    # 训练模型
    start_time = time.time()
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader_cifar, val_loader_cifar, criterion, optimizer, num_epochs, device
    )
    training_time = time.time() - start_time
    
    # 测试模型
    test_loss, test_acc = test_model(model, test_loader_cifar, criterion, device)
    
    # 存储结果
    results[opt_name] = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time
    }
    
    print(f"\n{opt_name} 最终结果:")
    print(f"  训练准确率: {train_accs[-1]:.2f}%")
    print(f"  验证准确率: {val_accs[-1]:.2f}%")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"  训练时间: {training_time:.2f}秒")
    
    # 保存模型
    torch.save(model.state_dict(), f'cifar10_{opt_name}_model.pth')

# ========== 结果对比和可视化 ==========
print("\n" + "=" * 60)
print("优化器对比结果总结")
print("=" * 60)

# 创建结果表格
print("\n对比表格:")
print("-" * 80)
print(f"{'Optimizer':<15} {'Train Acc(%)':<15} {'Val Acc(%)':<15} {'Test Acc(%)':<15} {'Test Loss':<12} {'Time(s)':<10}")
print("-" * 80)

for opt_name in results.keys():
    print(f"{opt_name:<15} {results[opt_name]['final_train_acc']:<15.2f} "
          f"{results[opt_name]['final_val_acc']:<15.2f} {results[opt_name]['test_acc']:<15.2f} "
          f"{results[opt_name]['test_loss']:<12.4f} {results[opt_name]['training_time']:<10.2f}")
print("-" * 80)

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 训练损失对比
for opt_name, data in results.items():
    axes[0, 0].plot(range(1, num_epochs+1), data['train_losses'], label=opt_name, marker='o', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Training Loss', fontsize=12)
axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 验证损失对比
for opt_name, data in results.items():
    axes[0, 1].plot(range(1, num_epochs+1), data['val_losses'], label=opt_name, marker='s', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Validation Loss', fontsize=12)
axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 训练准确率对比
for opt_name, data in results.items():
    axes[1, 0].plot(range(1, num_epochs+1), data['train_accs'], label=opt_name, marker='o', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Training Accuracy (%)', fontsize=12)
axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 验证准确率对比
for opt_name, data in results.items():
    axes[1, 1].plot(range(1, num_epochs+1), data['val_accs'], label=opt_name, marker='s', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Optimizer Comparison on CIFAR-10', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('optimizer_comparison_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ========== 任务3：学习率对比实验 ==========
print("\n" + "=" * 80)
print("任务3：学习率对比实验 - Adam优化器")
print("=" * 80)

# 定义要对比的学习率
learning_rates = [0.1, 0.01, 0.001]

# 存储学习率实验结果
lr_results = {}

print("\n" + "=" * 60)
print("开始学习率对比实验")
print("=" * 60)

for lr in learning_rates:
    print(f"\n{'='*60}")
    print(f"使用学习率: {lr}")
    print(f"{'='*60}")
    
    # 重置随机种子确保公平比较
    set_seed(42)
    
    # 创建新模型
    model = CIFAR10_CNN()
    
    # 创建Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
    start_time = time.time()
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader_cifar, val_loader_cifar, criterion, optimizer, num_epochs, device
    )
    training_time = time.time() - start_time
    
    # 测试模型
    test_loss, test_acc = test_model(model, test_loader_cifar, criterion, device)
    
    # 存储结果
    lr_results[lr] = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time
    }
    
    print(f"\n学习率 {lr} 最终结果:")
    print(f"  训练准确率: {train_accs[-1]:.2f}%")
    print(f"  验证准确率: {val_accs[-1]:.2f}%")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"  训练时间: {training_time:.2f}秒")
    
    # 保存模型
    torch.save(model.state_dict(), f'cifar10_Adam_lr{lr}_model.pth')

# ========== 学习率实验结果对比 ==========
print("\n" + "=" * 60)
print("学习率对比结果总结")
print("=" * 60)

# 创建结果表格
print("\n对比表格:")
print("-" * 90)
print(f"{'Learning Rate':<15} {'Train Acc(%)':<15} {'Val Acc(%)':<15} {'Test Acc(%)':<15} {'Test Loss':<12} {'Time(s)':<10}")
print("-" * 90)

for lr in learning_rates:
    print(f"{lr:<15} {lr_results[lr]['final_train_acc']:<15.2f} "
          f"{lr_results[lr]['final_val_acc']:<15.2f} {lr_results[lr]['test_acc']:<15.2f} "
          f"{lr_results[lr]['test_loss']:<12.4f} {lr_results[lr]['training_time']:<10.2f}")
print("-" * 90)

# 可视化学习率对比
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 训练损失对比
for lr, data in lr_results.items():
    axes[0, 0].plot(range(1, num_epochs+1), data['train_losses'], 
                    label=f'LR={lr}', marker='o', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Training Loss', fontsize=12)
axes[0, 0].set_title(f'Training Loss for Different Learning Rates (Adam)', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 验证损失对比
for lr, data in lr_results.items():
    axes[0, 1].plot(range(1, num_epochs+1), data['val_losses'], 
                    label=f'LR={lr}', marker='s', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Validation Loss', fontsize=12)
axes[0, 1].set_title(f'Validation Loss for Different Learning Rates (Adam)', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 训练准确率对比
for lr, data in lr_results.items():
    axes[1, 0].plot(range(1, num_epochs+1), data['train_accs'], 
                    label=f'LR={lr}', marker='o', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Training Accuracy (%)', fontsize=12)
axes[1, 0].set_title(f'Training Accuracy for Different Learning Rates (Adam)', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 验证准确率对比
for lr, data in lr_results.items():
    axes[1, 1].plot(range(1, num_epochs+1), data['val_accs'], 
                    label=f'LR={lr}', marker='s', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
axes[1, 1].set_title(f'Validation Accuracy for Different Learning Rates (Adam)', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Learning Rate Comparison on CIFAR-10 (Adam Optimizer)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('learning_rate_comparison_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ========== 任务4：卷积核可视化 ==========
print("\n" + "=" * 80)
print("任务4：卷积核可视化")
print("=" * 80)

# 加载训练好的模型（使用Adam优化器的模型作为示例）
model_vis = CIFAR10_CNN()
model_vis.load_state_dict(torch.load('cifar10_Adam_lr0.001_model.pth', map_location='cpu'))
model_vis.eval()

# 获取第一层卷积核
conv1_weights = model_vis.conv1.weight.data.cpu().numpy()
print(f"第一层卷积核形状: {conv1_weights.shape}")  # [out_channels, in_channels, height, width]
print(f"卷积核数量: {conv1_weights.shape[0]}")
print(f"每个卷积核尺寸: {conv1_weights.shape[2]}x{conv1_weights.shape[3]}")

# 归一化卷积核以便可视化
def normalize_kernel(kernel):
    """归一化卷积核到[0,1]范围"""
    kernel = kernel - kernel.min()
    if kernel.max() > 0:
        kernel = kernel / kernel.max()
    return kernel

# 显示前16个卷积核（3x3的卷积核，因为输入是RGB，每个卷积核有3个通道）
num_kernels_to_show = min(16, conv1_weights.shape[0])
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

for i in range(num_kernels_to_show):
    # 获取第i个卷积核（3个通道）
    kernel = conv1_weights[i]
    
    # 将3个通道合并显示（取RGB通道的平均或分别显示）
    # 方法1：显示每个通道的平均
    kernel_display = np.mean(kernel, axis=0)
    kernel_display = normalize_kernel(kernel_display)
    
    axes[i].imshow(kernel_display, cmap='coolwarm', interpolation='nearest')
    axes[i].set_title(f'Filter {i+1}', fontsize=10)
    axes[i].axis('off')
    
    # 添加颜色条
    if i == 0:
        plt.colorbar(axes[i].imshow(kernel_display, cmap='coolwarm'), ax=axes[i], fraction=0.046)

# 隐藏多余的子图
for i in range(num_kernels_to_show, len(axes)):
    axes[i].axis('off')

plt.suptitle('First Convolutional Layer Filters Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task4_conv1_filters.png', dpi=150, bbox_inches='tight')
plt.show()

# 单独显示8个最具代表性的卷积核
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()

# 选择8个不同的卷积核（间隔选取）
selected_indices = [0, 3, 5, 7, 9, 11, 13, 15]
for idx, i in enumerate(selected_indices[:8]):
    if i < conv1_weights.shape[0]:
        kernel = conv1_weights[i]
        kernel_display = np.mean(kernel, axis=0)
        kernel_display = normalize_kernel(kernel_display)
        
        axes[idx].imshow(kernel_display, cmap='gray', interpolation='nearest')
        axes[idx].set_title(f'Filter {i+1}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        # 分析卷积核特征
        kernel_std = np.std(kernel_display)
        kernel_mean = np.mean(kernel_display)
        if kernel_std > 0.3:
            feature = "Strong edges/direction"
        elif kernel_mean > 0.5:
            feature = "Blur/smoothing"
        else:
            feature = "Texture/detail"
        axes[idx].set_xlabel(feature, fontsize=8)

plt.suptitle('Selected Convolutional Filters (First Layer)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task4_selected_filters.png', dpi=150, bbox_inches='tight')
plt.show()


# ========== 任务5：Feature Map可视化 ==========
print("\n" + "=" * 80)
print("任务5：Feature Map可视化")
print("=" * 80)

# 选择一张测试图片
test_iter = iter(test_loader_cifar)
test_images, test_labels = next(test_iter)
test_image = test_images[0:1]  # 取第一张图片
true_label = test_labels[0].item()

# 显示原始图片
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
img_display = test_image[0] * std + mean
img_display = torch.clamp(img_display, 0, 1)
img_display = img_display.permute(1, 2, 0).numpy()

plt.figure(figsize=(6, 6))
plt.imshow(img_display)
plt.title(f'Input Image - True Label: {cifar_classes[true_label]}', fontsize=14, fontweight='bold')
plt.axis('off')
plt.savefig('task5_input_image.png', dpi=150, bbox_inches='tight')
plt.show()

# 获取第一层卷积输出
def get_feature_maps(model, input_image, layer_name='conv1'):
    """获取指定层的特征图"""
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # 注册hook
    handle = getattr(model, layer_name).register_forward_hook(hook_fn)
    
    # 前向传播
    with torch.no_grad():
        _ = model(input_image)
    
    # 移除hook
    handle.remove()
    
    return feature_maps[0]

# 获取第一层卷积的特征图
model_vis = model_vis.to(device)
test_image_tensor = test_image.to(device)
conv1_output = get_feature_maps(model_vis, test_image_tensor, 'conv1')
print(f"第一层卷积输出形状: {conv1_output.shape}")  # [batch, channels, height, width]

# 显示前16个特征图
num_feature_maps = min(16, conv1_output.shape[1])
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
axes = axes.ravel()

for i in range(num_feature_maps):
    # 获取第i个特征图
    feature_map = conv1_output[0, i, :, :].cpu().numpy()
    
    # 归一化以便显示
    feature_map_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
    
    axes[i].imshow(feature_map_norm, cmap='viridis', interpolation='nearest')
    axes[i].set_title(f'Feature Map {i+1}', fontsize=10)
    axes[i].axis('off')
    
    # 分析特征图激活区域
    activation_mean = feature_map_norm.mean()
    activation_std = feature_map_norm.std()
    if activation_mean > 0.3:
        activation_level = "High activation"
    elif activation_mean > 0.1:
        activation_level = "Medium activation"
    else:
        activation_level = "Low activation"
    axes[i].set_xlabel(f'{activation_level}', fontsize=8)

# 隐藏多余的子图
for i in range(num_feature_maps, len(axes)):
    axes[i].axis('off')

plt.suptitle(f'Feature Maps from First Convolutional Layer\nInput: {cifar_classes[true_label]}', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task5_feature_maps.png', dpi=150, bbox_inches='tight')
plt.show()

# 显示8个最具代表性的特征图
selected_fmaps = [0, 2, 4, 6, 8, 10, 12, 14]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, i in enumerate(selected_fmaps[:8]):
    if i < conv1_output.shape[1]:
        feature_map = conv1_output[0, i, :, :].cpu().numpy()
        feature_map_norm = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # 叠加在原始图像上显示
        axes[idx].imshow(img_display, alpha=0.5)
        im = axes[idx].imshow(feature_map_norm, cmap='hot', alpha=0.5, interpolation='nearest')
        axes[idx].set_title(f'Feature Map {i+1}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        # 分析特征图关注的区域
        high_activation = (feature_map_norm > 0.7).sum()
        if high_activation > 100:
            region = "Large area"
        elif high_activation > 30:
            region = "Medium area"
        else:
            region = "Small/isolated area"
        axes[idx].set_xlabel(f'Activation: {region}', fontsize=9)

plt.suptitle('Feature Maps Overlay on Original Image', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task5_feature_maps_overlay.png', dpi=150, bbox_inches='tight')
plt.show()


# ========== 任务6：错误分类样本分析 ==========
print("\n" + "=" * 80)
print("任务6：错误分类样本分析")
print("=" * 80)

# 获取所有错误分类的样本
model_vis = model_vis.to(device)
model_vis.eval()

misclassified_samples = []
with torch.no_grad():
    for images, labels in test_loader_cifar:
        images, labels = images.to(device), labels.to(device)
        outputs = model_vis(images)
        _, predicted = torch.max(outputs, 1)
        
        # 找出预测错误的样本
        incorrect_mask = (predicted != labels)
        if incorrect_mask.any():
            for i in range(len(images)):
                if incorrect_mask[i]:
                    misclassified_samples.append({
                        'image': images[i].cpu(),
                        'true_label': labels[i].item(),
                        'pred_label': predicted[i].item()
                    })

print(f"测试集总样本数: {len(test_dataset_cifar)}")
print(f"错误分类样本数: {len(misclassified_samples)}")
print(f"错误率: {len(misclassified_samples)/len(test_dataset_cifar)*100:.2f}%")

# 显示至少8个错误分类样本
num_samples = min(16, len(misclassified_samples))
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()

for i in range(num_samples):
    sample = misclassified_samples[i]
    image = sample['image']
    true_label = sample['true_label']
    pred_label = sample['pred_label']
    
    # 反归一化
    img_display = image * std + mean
    img_display = torch.clamp(img_display, 0, 1)
    img_display = img_display.permute(1, 2, 0).numpy()
    
    axes[i].imshow(img_display)
    
    # 设置标题颜色（红色表示错误）
    title = f'True: {cifar_classes[true_label]}\nPred: {cifar_classes[pred_label]}'
    axes[i].set_title(title, fontsize=10, color='red', fontweight='bold')
    axes[i].axis('off')

# 隐藏多余的子图
for i in range(num_samples, len(axes)):
    axes[i].axis('off')

plt.suptitle('Misclassified Samples Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task6_misclassified_samples.png', dpi=150, bbox_inches='tight')
plt.show()

# 统计混淆对
confusion_pairs = {}
for sample in misclassified_samples:
    pair = (cifar_classes[sample['true_label']], cifar_classes[sample['pred_label']])
    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

# 显示最常见的混淆对
print("\n最常见的混淆类别对:")
sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
for i, (pair, count) in enumerate(sorted_pairs[:10], 1):
    print(f"  {i}. {pair[0]} → {pair[1]}: {count}次")

# 分析错误原因
print("\n错误原因分析:")
print("=" * 60)
print("1. 最容易混淆的类别:")
confusion_counts = {}
for pair, count in confusion_pairs.items():
    confusion_counts[pair[0]] = confusion_counts.get(pair[0], 0) + count
sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
for category, count in sorted_confusions[:5]:
    print(f"   - {category}: 最常被误认为其他类别 ({count}次)")


# ========== 任务7：混淆矩阵（不使用sklearn） ==========
print("\n" + "=" * 80)
print("任务7：混淆矩阵")
print("=" * 80)

# 获取所有预测结果
all_labels = []
all_predictions = []

model_vis.eval()
with torch.no_grad():
    for images, labels in test_loader_cifar:
        images, labels = images.to(device), labels.to(device)
        outputs = model_vis(images)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 手动计算混淆矩阵
num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=np.int32)

for true_label, pred_label in zip(all_labels, all_predictions):
    cm[true_label, pred_label] += 1

print(f"混淆矩阵形状: {cm.shape}")
print(f"测试集总样本数: {len(all_labels)}")
print(f"正确分类数: {np.trace(cm)}")
print(f"错误分类数: {len(all_labels) - np.trace(cm)}")

# 归一化混淆矩阵（按行）
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 处理除零情况
cm_normalized = np.nan_to_num(cm_normalized)

# 绘制混淆矩阵
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# 原始计数混淆矩阵
im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xticks(np.arange(10))
ax1.set_yticks(np.arange(10))
ax1.set_xticklabels(cifar_classes, rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels(cifar_classes, fontsize=9)

# 添加数值标签
for i in range(10):
    for j in range(10):
        if cm[i, j] > 0:
            text = ax1.text(j, i, cm[i, j],
                           ha="center", va="center", 
                           color="white" if cm[i, j] > cm.max() / 2 else "black",
                           fontsize=8)

plt.colorbar(im1, ax=ax1)

# 归一化混淆矩阵（百分比）
im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=1)
ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xticks(np.arange(10))
ax2.set_yticks(np.arange(10))
ax2.set_xticklabels(cifar_classes, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(cifar_classes, fontsize=9)

# 添加百分比标签
for i in range(10):
    for j in range(10):
        if cm_normalized[i, j] > 0:
            text = ax2.text(j, i, f'{cm_normalized[i, j]*100:.1f}%',
                           ha="center", va="center", 
                           color="white" if cm_normalized[i, j] > 0.5 else "black",
                           fontsize=8)

plt.colorbar(im2, ax=ax2)

plt.suptitle('Confusion Matrix Analysis on CIFAR-10 Test Set', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task7_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# 计算各类别准确率
class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
# 处理除零情况
class_accuracy = np.nan_to_num(class_accuracy)

print("\n各类别分类准确率:")
for i, class_name in enumerate(cifar_classes):
    total_samples = cm[i].sum()
    correct = cm[i, i]
    print(f"  {class_name:12s}: {class_accuracy[i]:.2f}% ({correct}/{total_samples})")

# 找出最严重的混淆对（排除对角线）
max_confusion = 0
worst_pair = None
for i in range(10):
    for j in range(10):
        if i != j and cm_normalized[i, j] > max_confusion:
            max_confusion = cm_normalized[i, j]
            worst_pair = (cifar_classes[i], cifar_classes[j])

if worst_pair:
    print(f"\n3. 最严重的混淆:")
    print(f"   - 类别对: {worst_pair[0]} → {worst_pair[1]}")
    print(f"   - 混淆比例: {max_confusion*100:.2f}%")
    true_idx = cifar_classes.index(worst_pair[0])
    pred_idx = cifar_classes.index(worst_pair[1])
    print(f"   - 错误数量: {cm[true_idx, pred_idx]}")

# 找出前5个最严重的混淆对
confusion_list = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            confusion_list.append({
                'true': cifar_classes[i],
                'pred': cifar_classes[j],
                'count': cm[i, j],
                'rate': cm_normalized[i, j]
            })

confusion_list.sort(key=lambda x: x['rate'], reverse=True)
print("\n4. 前5个最严重的混淆对:")
for i, conf in enumerate(confusion_list[:5], 1):
    print(f"   {i}. {conf['true']} → {conf['pred']}: {conf['count']}次 ({conf['rate']*100:.1f}%)")

# 分析具体混淆模式
print("\n5. 详细混淆分析:")
# 统计每个类别最容易被误认为哪个类别
for i, class_name in enumerate(cifar_classes):
    if cm[i].sum() > 0:
        misclassifications = [(cifar_classes[j], cm[i, j]) for j in range(10) if j != i and cm[i, j] > 0]
        if misclassifications:
            most_confused = max(misclassifications, key=lambda x: x[1])
            print(f"   {class_name:12s} 最常被误认为: {most_confused[0]:12s} ({most_confused[1]}次)")
