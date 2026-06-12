"""
MediaPipe Pose 骨架提取 - 最终工作版本
Python 3.9 + mediapipe 0.10.9
33个关键点 × 4个特征 = 132维
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
DATA_ROOT = "/home/wxf81/作业12/作业15/badminton_stroke_video"
OUTPUT_DIR = "/home/wxf81/作业12/作业15/processed_data"
TARGET_FRAMES = 30
KEYPOINT_DIM = 132  # 33 × 4
TEST_SIZE = 0.2
RANDOM_SEED = 42

CLASS_NAMES = [
    "backhand_drive",
    "backhand_net_shot",
    "forehand_clear",
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=轻量, 1=中等, 2=高精度
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_pose_from_video(video_path):
    """从视频中提取骨架序列"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100
    
    # 采样帧索引
    sample_indices = np.linspace(0, total_frames - 1, TARGET_FRAMES, dtype=int)
    
    all_frames_landmarks = []
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            all_frames_landmarks.append([0.0] * KEYPOINT_DIM)
            continue
        
        # 转换颜色空间 BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # 提取 33 个关键点的 x, y, z, visibility
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            all_frames_landmarks.append(landmarks)
        else:
            # 未检测到人体，添加零向量
            all_frames_landmarks.append([0.0] * KEYPOINT_DIM)
    
    cap.release()
    
    if len(all_frames_landmarks) == 0:
        return None
    
    skeleton_seq = np.array(all_frames_landmarks, dtype=np.float32)
    
    # 归一化
    skeleton_seq = normalize_pose(skeleton_seq)
    
    return skeleton_seq

def normalize_pose(skeleton_seq):
    """
    归一化骨架：
    1. 以左右髋部中心为原点
    2. 以肩宽为尺度
    """
    T = skeleton_seq.shape[0]
    normalized = np.zeros_like(skeleton_seq)
    
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    
    for t in range(T):
        frame = skeleton_seq[t].reshape(33, 4)
        
        # 检查是否检测到有效关键点
        left_hip = frame[LEFT_HIP_IDX, :2]
        right_hip = frame[RIGHT_HIP_IDX, :2]
        
        if np.all(left_hip == 0) and np.all(right_hip == 0):
            normalized[t] = skeleton_seq[t]
            continue
        
        # 髋部中心
        hip_center = (left_hip + right_hip) / 2
        
        # 肩宽
        left_shoulder = frame[LEFT_SHOULDER_IDX, :2]
        right_shoulder = frame[RIGHT_SHOULDER_IDX, :2]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        shoulder_width = max(shoulder_width, 1e-6)
        
        # 归一化
        frame[:, 0] = (frame[:, 0] - hip_center[0]) / shoulder_width
        frame[:, 1] = (frame[:, 1] - hip_center[1]) / shoulder_width
        frame[:, 2] = frame[:, 2] / shoulder_width
        
        normalized[t] = frame.reshape(-1)
    
    return normalized

def process_dataset():
    """处理所有视频"""
    print("开始提取骨架特征...")
    
    X, y = [], []
    failed = []
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_ROOT, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: {class_dir} 不存在")
            continue
        
        video_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        class_id = CLASS_TO_IDX[class_name]
        print(f"\n处理 {class_name} (id={class_id}): {len(video_files)} 个视频")
        
        success = 0
        for video_file in tqdm(video_files, desc=class_name):
            video_path = os.path.join(class_dir, video_file)
            
            try:
                skeleton = extract_pose_from_video(video_path)
                
                if skeleton is not None and skeleton.shape == (TARGET_FRAMES, KEYPOINT_DIM):
                    X.append(skeleton)
                    y.append(class_id)
                    success += 1
                else:
                    failed.append(video_file)
            except Exception as e:
                failed.append(video_file)
                if len(failed) <= 5:
                    print(f"\n  错误: {video_file} - {str(e)}")
        
        print(f"  成功: {success}/{len(video_files)}")
    
    if len(X) == 0:
        return np.array([]), np.array([])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"\n{'='*50}")
    print(f"总成功: {len(X)} 个样本")
    print(f"总失败: {len(failed)} 个")
    print(f"X shape: {X.shape}")  # (N, 30, 132)
    print(f"y shape: {y.shape}")
    
    # 类别分布
    unique, counts = np.unique(y, return_counts=True)
    print("\n类别分布:")
    for u, c in zip(unique, counts):
        print(f"  {u} ({CLASS_NAMES[u]}): {c}")
    
    return X, y

def split_and_save(X, y):
    """划分并保存"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    
    label_map = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    
    config = {
        "target_frames": TARGET_FRAMES,
        "keypoint_dim": KEYPOINT_DIM,
        "num_keypoints": 33,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test))
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n数据已保存到: {OUTPUT_DIR}")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")

def main():
    print("="*60)
    print("MediaPipe Pose 骨架提取")
    print(f"MediaPipe 版本: {mp.__version__}")
    print("33个关键点 × 4特征 = 132维")
    print("="*60)
    
    if not os.path.exists(DATA_ROOT):
        print(f"错误: 数据路径不存在: {DATA_ROOT}")
        return
    
    X, y = process_dataset()
    
    if len(X) == 0:
        print("\n错误: 没有成功处理任何视频")
        return
    
    split_and_save(X, y)
    print("\n预处理完成！")
    
    # 关闭 pose 资源
    pose.close()

if __name__ == "__main__":
    main()