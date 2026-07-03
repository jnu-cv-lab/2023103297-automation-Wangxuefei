import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, util
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -------------------------- 自行实现模块：全局直方图均衡 --------------------------
def my_global_hist_eq(img):
    """
    自行实现全局直方图均衡化
    输入：灰度图像 (0-255)
    输出：均衡化后的图像
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算直方图
    hist, bins = np.histogram(img.flatten(), 256, (0, 256))
    # 计算累积分布函数
    cdf = hist.cumsum()
    # 归一化
    cdf_normalized = cdf * 255 / cdf[-1]
    # 映射
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    return img_equalized.reshape(img.shape).astype(np.uint8)

# -------------------------- 定量评价指标 --------------------------
def image_entropy(img):
    """信息熵：衡量图像信息量，值越大信息越丰富"""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    # 避免log(0)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def avg_gradient(img):
    """平均梯度：反映图像清晰度，值越大边缘越锐利"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    return np.mean(grad)

# -------------------------- 图像增强方法封装 --------------------------
def global_hist_eq(img):
    """全局直方图均衡化（调用 OpenCV 接口）"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    """CLAHE"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe_obj.apply(img)

def gaussian_filter(img, ksize=5, sigma=1.0):
    """高斯滤波（OpenCV接口）"""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def median_filter(img, ksize=3):
    """中值滤波（OpenCV接口）"""
    return cv2.medianBlur(img, ksize)

def laplacian_sharpen(img, alpha=1.0):
    """拉普拉斯锐化：原图 + alpha * 拉普拉斯"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    sharpened = img.astype(np.float32) + alpha * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def filter_then_equalize(img, filter_func, **filter_kwargs):
    """滤波→均衡"""
    filtered = filter_func(img, **filter_kwargs)
    return my_global_hist_eq(filtered)   # 使用自行实现的均衡

def equalize_then_filter(img, filter_func, **filter_kwargs):
    """均衡→滤波"""
    equalized = my_global_hist_eq(img)   # 使用自行实现的均衡
    return filter_func(equalized, **filter_kwargs)

# -------------------------- 准备三幅图像 --------------------------
# 1. 低对比度图像（使用skimage的moon图并降低对比度）
img_low_contrast = data.moon()
# 人为降低对比度：缩放像素范围到[80,180]
img_low_contrast = exposure.rescale_intensity(img_low_contrast, in_range='image', out_range=(80,180)) # type: ignore
img_low_contrast = img_low_contrast.astype(np.uint8)

# 2. 高斯噪声图像（使用camera图添加高斯噪声）
img_clean = data.camera()
img_gaussian = util.random_noise(img_clean, mode='gaussian', var=0.02)
img_gaussian = (img_gaussian * 255).astype(np.uint8)

# 3. 椒盐噪声图像（使用chelsea图添加椒盐噪声）
img_clean2 = data.chelsea()
# 转为灰度便于处理（也可以保留彩色，但为统一，转灰度）
if len(img_clean2.shape) == 3:
    img_clean2 = cv2.cvtColor(img_clean2, cv2.COLOR_RGB2GRAY)
img_saltpepper = util.random_noise(img_clean2, mode='s&p', amount=0.05)
img_saltpepper = (img_saltpepper * 255).astype(np.uint8)

# 将图像存入字典
images = {
    'Low Contrast': img_low_contrast,
    'Gaussian Noise': img_gaussian,
    'Salt & Pepper Noise': img_saltpepper
}

# -------------------------- 定义所有要比较的方法及参数 --------------------------
methods = {
    'Original': lambda img: img,
    'Global Hist Eq (OpenCV)': global_hist_eq,
    'Global Hist Eq (My Impl)': my_global_hist_eq,   # 自行实现版本
    'CLAHE (clip=2.0)': lambda img: clahe(img, clip_limit=2.0),
    'CLAHE (clip=4.0)': lambda img: clahe(img, clip_limit=4.0),
    'Mean Filter (3x3)': lambda img: cv2.blur(img, (3,3)),
    'Mean Filter (5x5)': lambda img: cv2.blur(img, (5,5)),
    'Gaussian Filter (5x5, σ=1)': lambda img: gaussian_filter(img, 5, 1.0),
    'Gaussian Filter (7x7, σ=2)': lambda img: gaussian_filter(img, 7, 2.0),
    'Median Filter (3x3)': lambda img: median_filter(img, 3),
    'Median Filter (5x5)': lambda img: median_filter(img, 5),
    'Laplacian Sharpen (α=1)': lambda img: laplacian_sharpen(img, alpha=1.0),
    'Laplacian Sharpen (α=2)': lambda img: laplacian_sharpen(img, alpha=2.0),
    'Filter→Eq (Mean3→Eq)': lambda img: filter_then_equalize(img, cv2.blur, ksize=(3,3)),
    'Eq→Filter (Eq→Mean3)': lambda img: equalize_then_filter(img, cv2.blur, ksize=(3,3))
}

# -------------------------- 对每幅图像进行实验 --------------------------
for img_name, img in images.items():
    print(f"\n========== 图像: {img_name} ==========")
    # 存储结果
    results = {}
    metrics = {}

    for method_name, func in methods.items():
        # 应用方法
        processed = func(img)
        results[method_name] = processed
        # 计算指标
        ent = image_entropy(processed)
        grad = avg_gradient(processed)
        metrics[method_name] = (ent, grad)
        print(f"{method_name:30s} | Entropy: {ent:.3f} | AvgGrad: {grad:.3f}")

        # 保存单个结果图像（文件名安全化）
        safe_method_name = method_name.replace(' ', '_').replace('σ', 'sigma').replace('α', 'alpha').replace('(', '_').replace(')', '_')
        out_img_path = os.path.join('/home/wxf81/实验4/output', f"{img_name}_{safe_method_name}.png")
        cv2.imwrite(out_img_path, processed)
        print(f"保存图像: {out_img_path}")

    # 保存定量指标到文本文件
    metrics_path = os.path.join('/home/wxf81/实验4/output', f"{img_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Metrics for {img_name}\n")
        f.write(f"{'Method':30s} | Entropy | AvgGrad\n")
        f.write("-" * 60 + "\n")
        for method_name, (ent, grad) in metrics.items():
            f.write(f"{method_name:30s} | {ent:.3f} | {grad:.3f}\n")
    print(f"保存指标: {metrics_path}")

    # 可视化并保存组合大图（包含所有结果及直方图）
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 3*n_methods))
    fig.suptitle(f'Image: {img_name}', fontsize=16)

    for i, (method_name, processed) in enumerate(results.items()):
        # 显示结果图像
        ax_img = axes[i, 0]
        ax_img.imshow(processed, cmap='gray')
        ax_img.set_title(method_name)
        ax_img.axis('off')
        # 显示直方图
        ax_hist = axes[i, 1]
        ax_hist.hist(processed.ravel(), bins=256, range=(0,256), color='gray', alpha=0.7)
        ax_hist.set_title('Histogram')
        ax_hist.set_xlim(0, 255)

    plt.tight_layout()
    combined_path = os.path.join('/home/wxf81/实验4/output', f"{img_name}_combined.png")
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"保存组合图: {combined_path}")
    plt.show()
    