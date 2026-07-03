import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import os

def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, '..', 'images', 'input', 'test.jpg')
    image_path = os.path.normpath(image_path)
    
    print(f"读取图片: {image_path}")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        return
    
    # 读取图片
    img = cv2.imread(image_path)
    
    if img is None:
        print("错误: 无法读取图片")
        return
    
    # 输出图像信息
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    
    print("\n" + "="*50)
    print("✅ 图片读取成功！")
    print(f"图像尺寸: {width} x {height}")
    print(f"图像宽度: {width} 像素")
    print(f"图像高度: {height} 像素")
    print(f"通道数: {channels}")
    print(f"数据类型: {img.dtype}")
    print("="*50)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ========== 任务6：使用NumPy裁剪图像左上角区域 ==========
    # 定义裁剪区域大小
    crop_size = 300  # 裁剪300x300像素区域
    
    # 确保裁剪区域不超过图像边界
    crop_height = min(crop_size, height)
    crop_width = min(crop_size, width)
    
    # 使用NumPy切片操作裁剪图像左上角区域
    # 对于彩色图像，裁剪 [0:crop_height, 0:crop_width, :]
    # 对于灰度图像，裁剪 [0:crop_height, 0:crop_width]
    if len(img.shape) == 3:
        cropped = img[0:crop_height, 0:crop_width, :]
    else:
        cropped = img[0:crop_height, 0:crop_width]
    
    print(f"\n📐 裁剪信息:")
    print(f"裁剪区域大小: {crop_width} x {crop_height} 像素")
    print(f"裁剪区域位置: 左上角 (0, 0)")
    print(f"裁剪后图像形状: {cropped.shape}")
    
    # 保存裁剪后的图像
    crop_output_path = os.path.join(current_dir, '..', 'images', 'output', 'cropped_top_left.jpg')
    cv2.imwrite(crop_output_path, cropped)
    print(f"\n✅ 裁剪图像已保存: {crop_output_path}")
    
    # ========== 显示所有图像 ==========
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Original\n{width}x{height}')
    plt.axis('off')
    
    # 显示灰度图像
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title(f'Grayscale\n{width}x{height}')
    plt.axis('off')
    
    # 显示裁剪后的图像
    plt.subplot(1, 3, 3)
    if len(cropped.shape) == 3:
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(cropped, cmap='gray')
    plt.title(f'Cropped Top-Left\n{crop_width}x{crop_height}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存灰度图（保留原有功能）
    output_path = os.path.join(current_dir, '..', 'images', 'output', 'grayscale.jpg')
    cv2.imwrite(output_path, gray)
    print(f"\n灰度图已保存: {output_path}")

if __name__ == "__main__":
    main()