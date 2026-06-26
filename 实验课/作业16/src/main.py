import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
# 棋盘格内角点数量 (9x6)
CHECKERBOARD = (9, 6)
# 方格边长 (单位: mm)
SQUARE_SIZE = 25

# 图片路径
IMAGE_DIR = "/home/wxf81/作业12/作业16/images"
OUTPUT_DIR = "/home/wxf81/作业12/作业16/output"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 步骤1: 定义三维角点坐标 ====================
def get_object_points(checkerboard, square_size):
    """生成标定板坐标系中的三维角点坐标"""
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp = objp * square_size  # 单位: mm
    return objp

# ==================== 步骤2: 读取图片并检测角点 ====================
def detect_corners(image_paths, checkerboard, objp):
    """检测所有图片中的棋盘格角点"""
    objpoints = []  # 三维世界坐标
    imgpoints = []  # 二维图像坐标
    img_list = []
    failed_images = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for i, fname in enumerate(image_paths):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告: 无法读取图片 {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        
        if ret:
            objpoints.append(objp)
            # 亚像素精度优化
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # 绘制角点
            img_copy = img.copy()
            cv2.drawChessboardCorners(img_copy, checkerboard, corners2, ret)
            # 保存角点检测结果图
            output_path = os.path.join(OUTPUT_DIR, f"corners_{i+1:03d}.jpg")
            cv2.imwrite(output_path, img_copy)
            
            img_list.append((img, gray, fname))
            print(f"✓ 图片 {i+1}: 角点检测成功")
        else:
            failed_images.append(fname)
            print(f"✗ 图片 {i+1}: 角点检测失败")
    
    print(f"\n成功检测: {len(imgpoints)} 张, 失败: {len(failed_images)} 张")
    return objpoints, imgpoints, img_list, failed_images

# ==================== 步骤3: 相机标定 ====================
def calibrate_camera(objpoints, imgpoints, img_shape):
    """执行相机标定"""
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape[::-1], None, None
    )
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    return ret, mtx, dist, rvecs, tvecs, mean_error

# ==================== 步骤4: 去畸变处理 ====================
def undistort_images(img_list, mtx, dist, output_dir):
    """对图片进行去畸变处理"""
    undistorted_results = []
    
    for i, (img, gray, fname) in enumerate(img_list[:5]):  # 处理前5张
        h, w = img.shape[:2]
        
        # 方法1: 不裁剪，保持原始尺寸
        dst = cv2.undistort(img, mtx, dist, None)
        
        # 方法2: 获取最优相机矩阵并裁剪
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst_cropped = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w_crop, h_crop = roi
        dst_cropped = dst_cropped[y:y+h_crop, x:x+w_crop]
        
        # 保存原图（用于对比）
        base_name = os.path.basename(fname)
        name, ext = os.path.splitext(base_name)
        
        # 保存去畸变后的图片（不裁剪）
        output_path = os.path.join(output_dir, f"undistorted_{name}{ext}")
        cv2.imwrite(output_path, dst)
        
        # 保存裁剪后的去畸变图片
        output_cropped_path = os.path.join(output_dir, f"undistorted_cropped_{name}{ext}")
        cv2.imwrite(output_cropped_path, dst_cropped)
        
        # 创建对比图 - 统一尺寸
        # 如果需要保持相同高度，调整dst尺寸
        if dst.shape[0] != img.shape[0]:
            dst_resized = cv2.resize(dst, (w, h))
        else:
            dst_resized = dst
            
        # 水平拼接对比
        compare = np.hstack([img, dst_resized])
        compare_path = os.path.join(output_dir, f"compare_{i+1:03d}.jpg")
        cv2.imwrite(compare_path, compare)
        
        # 创建三图对比（原始 + 去畸变 + 裁剪去畸变）
        # 确保三张图高度一致
        h_compare = min(img.shape[0], dst.shape[0], dst_cropped.shape[0])
        img_resized = cv2.resize(img, (int(img.shape[1] * h_compare / img.shape[0]), h_compare))
        dst_resized = cv2.resize(dst, (int(dst.shape[1] * h_compare / dst.shape[0]), h_compare))
        dst_cropped_resized = cv2.resize(dst_cropped, (int(dst_cropped.shape[1] * h_compare / dst_cropped.shape[0]), h_compare))
        
        compare_three = np.hstack([img_resized, dst_resized, dst_cropped_resized])
        compare_three_path = os.path.join(output_dir, f"compare_three_{i+1:03d}.jpg")
        cv2.imwrite(compare_three_path, compare_three)
        
        undistorted_results.append((img, dst, dst_cropped, compare))
        print(f"✓ 去畸变处理: {base_name}")
    
    return undistorted_results

# ==================== 步骤5: 可视化结果 ====================
def visualize_results(output_dir):
    """可视化标定结果"""
    try:
        # 找到对比图
        compare_files = glob.glob(os.path.join(output_dir, "compare_*.jpg"))
        if compare_files:
            # 读取前两张对比图
            for i, f in enumerate(compare_files[:2]):
                img = cv2.imread(f)
                if img is not None:
                    plt.figure(figsize=(15, 5))
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(f"去畸变对比图 {i+1} (左: 原始, 右: 去畸变)")
                    plt.axis('off')
                    plt.savefig(os.path.join(output_dir, f"visualization_{i+1}.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    
        # 找到三图对比
        compare_three_files = glob.glob(os.path.join(output_dir, "compare_three_*.jpg"))
        for i, f in enumerate(compare_three_files[:1]):
            img = cv2.imread(f)
            if img is not None:
                plt.figure(figsize=(20, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("三图对比 (左: 原始, 中: 去畸变, 右: 裁剪去畸变)")
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"visualization_three_{i+1}.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
        print(f"✓ 可视化结果已保存")
    except Exception as e:
        print(f"可视化生成警告: {e}")

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("棋盘格相机标定程序")
    print("=" * 60)
    
    # 获取所有图片
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    image_paths.sort()
    
    if len(image_paths) == 0:
        print(f"错误: 在 {IMAGE_DIR} 中未找到图片")
        return
    
    print(f"\n找到 {len(image_paths)} 张图片\n")
    
    # 生成三维角点坐标
    objp = get_object_points(CHECKERBOARD, SQUARE_SIZE)
    
    # 检测角点
    objpoints, imgpoints, img_list, failed_images = detect_corners(
        image_paths, CHECKERBOARD, objp
    )
    
    if len(imgpoints) < 5:
        print("\n错误: 成功检测的角点图片少于5张，无法进行标定")
        return
    
    # 执行标定
    print("\n执行相机标定...")
    img_shape = img_list[0][0].shape[:2]
    ret, mtx, dist, rvecs, tvecs, mean_error = calibrate_camera(
        objpoints, imgpoints, img_shape
    )
    
    print(f"\n标定完成!")
    print(f"重投影误差: {mean_error:.4f} pixels")
    print(f"\n相机内参矩阵 K:\n{mtx}")
    print(f"\n畸变参数 D:\n{dist.ravel()}")
    
    # 去畸变处理
    print("\n进行去畸变处理...")
    undistort_images(img_list, mtx, dist, OUTPUT_DIR)
    
    # 可视化
    print("\n生成可视化结果...")
    visualize_results(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("程序运行完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()