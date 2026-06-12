"""
Skeleton Transformer 训练代码
用于羽毛球击球动作识别
基于 MediaPipe 33关键点骨架序列
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
DATA_DIR = "/home/wxf81/作业12/作业15/processed_data"
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# 模型参数
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.3

NUM_CLASSES = 6
TARGET_FRAMES = 30
INPUT_DIM = 132
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")
print(f"数据路径: {DATA_DIR}")

# ==================== 数据集类 ====================
class SkeletonDataset(Dataset):
    """骨架序列数据集"""
    
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            # 时间掩码增强
            if np.random.random() > 0.5:
                X = self.time_masking(X)
            # 添加噪声
            if np.random.random() > 0.5:
                X = self.add_noise(X)
        
        return X, y
    
    def time_masking(self, X, mask_ratio=0.1):
        """时间维度掩码"""
        T = X.shape[0]
        mask_len = max(1, int(T * mask_ratio))
        start = np.random.randint(0, T - mask_len)
        X[start:start+mask_len] = 0
        return X
    
    def add_noise(self, X, noise_std=0.01):
        """添加高斯噪声"""
        noise = torch.randn_like(X) * noise_std
        return X + noise

# ==================== 位置编码 ====================
class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    
    def __init__(self, d_model, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

# ==================== Transformer 模型 ====================
class SkeletonTransformer(nn.Module):
    """
    骨架序列 Transformer 分类模型
    输入: (batch, seq_len, input_dim) -> (batch, num_classes)
    """
    
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, max_len=TARGET_FRAMES):
        super(SkeletonTransformer, self).__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        返回: (batch, num_classes)
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 全局平均池化（沿时间维度）
        x = x.mean(dim=1)
        
        # 分类
        logits = self.classifier(x)
        
        return logits

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

# ==================== 数据预处理 ====================
def preprocess_data(X_train, X_test):
    """标准化数据"""
    original_shape_train = X_train.shape
    original_shape_test = X_test.shape
    
    X_train_flat = X_train.reshape(-1, INPUT_DIM)
    X_test_flat = X_test.reshape(-1, INPUT_DIM)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(original_shape_train)
    X_test = X_test_flat.reshape(original_shape_test)
    
    return X_train, X_test, scaler

# ==================== 可视化 ====================
def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_dir):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.1667, color='gray', linestyle='--', label='Random Guess (16.67%)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"训练曲线已保存: {os.path.join(save_dir, 'training_curves.png')}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted', ylabel='True')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {os.path.join(save_dir, 'confusion_matrix.png')}")
    
    return cm

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("羽毛球击球动作识别 - Skeleton Transformer 训练")
    print("="*60)
    print(f"设备: {DEVICE}")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    
    # 检查数据文件是否存在
    required_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "label_map.json"]
    for f in required_files:
        file_path = os.path.join(DATA_DIR, f)
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            print("请先运行预处理脚本")
            return
    
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    with open(os.path.join(DATA_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"类别: {class_names}")
    
    # 打印类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n训练集类别分布:")
    for u, c in zip(unique, counts):
        print(f"  {class_names[u]}: {c}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    X_train, X_test, scaler = preprocess_data(X_train, X_test)
    print(f"数据标准化完成")
    print(f"训练集范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    # 3. 创建 DataLoader
    train_dataset = SkeletonDataset(X_train, y_train, augment=True)
    test_dataset = SkeletonDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 4. 创建模型
    print("\n3. 创建模型...")
    model = SkeletonTransformer().to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print(f"\n模型配置:")
    print(f"  Input: (batch, {TARGET_FRAMES}, {INPUT_DIM})")
    print(f"  d_model: {D_MODEL}")
    print(f"  Transformer层数: {NUM_LAYERS}")
    print(f"  注意力头数: {NHEAD}")
    print(f"  FFN维度: {DIM_FEEDFORWARD}")
    print(f"  Dropout: {DROPOUT}")
    
    # 5. 训练配置
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print(f"\n4. 开始训练...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    print(f"   Label Smoothing: 0.1")
    
    # 6. 训练循环
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(DATA_DIR, 'best_model.pth'))
            print(f"✓ 保存最佳模型 (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停！验证准确率 {patience} 轮未提升")
                break
    
    # 7. 最终评估
    print("\n" + "="*60)
    print("5. 最终评估")
    print("="*60)
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(DATA_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最佳模型 (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n最终测试结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # 8. 可视化
    print("\n6. 生成可视化...")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, DATA_DIR)
    plot_confusion_matrix(test_labels, test_preds, class_names, DATA_DIR)
    
    # 9. 保存配置
    config = {
        "batch_size": BATCH_SIZE,
        "epochs": epoch,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc)
    }
    with open(os.path.join(DATA_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"测试准确率: {test_acc*100:.2f}%")
    print(f"模型保存到: {DATA_DIR}/best_model.pth")
    print("="*60)

if __name__ == "__main__":
    main()