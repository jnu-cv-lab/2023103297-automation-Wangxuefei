"""
测试与推理代码 - 与训练模型结构一致
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ==================== 配置参数（必须与训练时一致） ====================
DATA_DIR = "/home/wxf81/作业12/作业15/processed_data"
MODEL_PATH = os.path.join(DATA_DIR, "best_model_optimized.pth")
TARGET_FRAMES = 30
INPUT_DIM = 132
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数（必须与训练时一致）
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.3

print(f"使用设备: {DEVICE}")

# ==================== 模型定义（与训练完全一致） ====================
class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class ImprovedSkeletonTransformer(nn.Module):
    """改进版骨架序列 Transformer - 与训练一致"""
    
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, max_len=TARGET_FRAMES):
        super(ImprovedSkeletonTransformer, self).__init__()
        
        # 输入投影 + BatchNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(max_len),
            nn.Dropout(dropout)
        )
        
        # 可学习位置编码
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
        
        # 注意力池化
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # 注意力池化
        batch_size = x.shape[0]
        query = torch.randn(1, 1, D_MODEL).repeat(batch_size, 1, 1).to(x.device)
        x_attended, _ = self.attention_pool(query, x, x)
        x = x_attended.squeeze(1)
        
        # LayerNorm
        x = self.norm(x)
        
        # 分类
        logits = self.classifier(x)
        
        return logits

# ==================== MediaPipe 姿态提取 ====================
mp_pose = mp.solutions.pose

def extract_pose_from_video(video_path, target_frames=TARGET_FRAMES):
    """从视频中提取骨架序列"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100
    
    sample_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    all_landmarks = []
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                all_landmarks.append([0.0] * 132)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                all_landmarks.append(landmarks)
            else:
                all_landmarks.append([0.0] * 132)
    
    cap.release()
    
    if len(all_landmarks) == 0:
        return None
    
    skeleton_seq = np.array(all_landmarks, dtype=np.float32)
    skeleton_seq = normalize_pose(skeleton_seq)
    
    return skeleton_seq

def normalize_pose(skeleton_seq):
    """归一化骨架"""
    T = skeleton_seq.shape[0]
    normalized = np.zeros_like(skeleton_seq)
    
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    
    for t in range(T):
        frame = skeleton_seq[t].reshape(33, 4)
        
        left_hip = frame[LEFT_HIP, :2]
        right_hip = frame[RIGHT_HIP, :2]
        
        if np.all(left_hip == 0) and np.all(right_hip == 0):
            normalized[t] = skeleton_seq[t]
            continue
        
        hip_center = (left_hip + right_hip) / 2
        left_shoulder = frame[LEFT_SHOULDER, :2]
        right_shoulder = frame[RIGHT_SHOULDER, :2]
        shoulder_width = max(np.linalg.norm(left_shoulder - right_shoulder), 1e-6)
        
        frame[:, 0] = (frame[:, 0] - hip_center[0]) / shoulder_width
        frame[:, 1] = (frame[:, 1] - hip_center[1]) / shoulder_width
        frame[:, 2] = frame[:, 2] / shoulder_width
        
        normalized[t] = frame.reshape(-1)
    
    return normalized

# ==================== 测试集评估 ====================
def evaluate_test_set(model, device):
    """在测试集上评估模型"""
    print("\n" + "="*60)
    print("测试集评估")
    print("="*60)
    
    # 加载测试数据
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    with open(os.path.join(DATA_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    
    print(f"测试集大小: {len(X_test)}")
    print(f"X_test shape: {X_test.shape}")
    
    # 转换为 tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = outputs.argmax(dim=1)
    
    preds_np = preds.cpu().numpy()
    y_test_np = y_test
    
    # 计算指标
    accuracy = accuracy_score(y_test_np, preds_np)
    print(f"\n测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n分类报告:")
    print(classification_report(y_test_np, preds_np, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test_np, preds_np)
    print("\n混淆矩阵:")
    print(" " * 15 + "".join([f"{name[:8]:>8}" for name in class_names]))
    for i, name in enumerate(class_names):
        row_str = f"{name[:12]:<12}"
        for j in range(len(class_names)):
            row_str += f"{cm[i, j]:>8}"
        print(row_str)
    
    return accuracy, class_names, cm

# ==================== 单视频推理 ====================
def inference_single_video(model, video_path, class_names, device):
    """对单个视频进行推理"""
    print(f"\n处理视频: {os.path.basename(video_path)}")
    
    # 提取骨架特征
    skeleton_seq = extract_pose_from_video(video_path)
    
    if skeleton_seq is None:
        print("错误: 无法提取视频特征")
        return None
    
    print(f"特征提取成功: {skeleton_seq.shape}")
    
    # 转换为 tensor 并推理
    input_tensor = torch.FloatTensor(skeleton_seq).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # 显示结果
    print("\n" + "="*40)
    print("推理结果")
    print("="*40)
    print(f"预测类别: {class_names[pred_class]}")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # 显示所有类别的概率
    print("\n所有类别概率:")
    for i, name in enumerate(class_names):
        prob = probs[0, i].item()
        bar = "█" * int(prob * 50)
        print(f"  {name:20s}: {prob:.4f} {bar}")
    
    return {
        "pred_class": pred_class,
        "pred_class_name": class_names[pred_class],
        "confidence": confidence,
        "all_probs": probs[0].cpu().numpy()
    }

# ==================== 加载模型 ====================
def load_model(model_path, device):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本")
        return None
    
    # 创建模型（与训练一致）
    model = ImprovedSkeletonTransformer().to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n模型加载成功！")
    print(f"  训练 epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  验证准确率: {checkpoint.get('val_acc', 0):.4f}")
    
    return model

# ==================== 可视化 ====================
def visualize_prediction(video_path, class_names, pred_class, confidence, save_path=None):
    """可视化推理结果"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("无法读取视频帧")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(frame_rgb)
    ax.axis('off')
    
    title = f"Predicted: {class_names[pred_class]}\nConfidence: {confidence:.2%}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']
    color = colors[pred_class % len(colors)]
    
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    
    plt.show()

# ==================== 主函数 ====================
def main():
    print("="*60)
    print("羽毛球击球动作识别 - 测试与推理")
    print(f"模型参数: d_model={D_MODEL}, nhead={NHEAD}, num_layers={NUM_LAYERS}")
    print("="*60)
    
    # 1. 加载模型
    print("\n加载模型...")
    model = load_model(MODEL_PATH, DEVICE)
    if model is None:
        return
    
    # 2. 加载标签映射
    label_path = os.path.join(DATA_DIR, "label_map.json")
    if not os.path.exists(label_path):
        print(f"错误: 标签文件不存在: {label_path}")
        return
    
    with open(label_path, "r") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    print(f"类别: {class_names}")
    
    # 3. 测试集评估
    evaluate_test_set(model, DEVICE)
    
    # 4. 单视频推理示例
    print("\n" + "="*60)
    print("单视频推理示例")
    print("="*60)
    
    # 寻找一个测试视频
    test_video = None
    for class_name in class_names:
        class_dir = os.path.join("/home/wxf81/作业12/作业15/badminton_stroke_video", class_name)
        if os.path.exists(class_dir):
            videos = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            if videos:
                test_video = os.path.join(class_dir, videos[0])
                break
    
    if test_video and os.path.exists(test_video):
        result = inference_single_video(model, test_video, class_names, DEVICE)
        
        if result:
            vis_path = os.path.join(DATA_DIR, "inference_result.png")
            visualize_prediction(test_video, class_names, 
                               result['pred_class'], result['confidence'], vis_path)
    else:
        print("没有找到测试视频，跳过单视频推理")
    
    print("\n" + "="*60)
    print("测试与推理完成！")
    print("="*60)

if __name__ == "__main__":
    main()