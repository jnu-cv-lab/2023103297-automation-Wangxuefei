import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from scipy.fftpack import dct, idct

# ----------------------------- 1. 读取灰度图像 -----------------------------
img_color = data.astronaut()
img = color.rgb2gray(img_color)
img = (img * 255).astype(np.uint8)

print("使用图片: skimage.data.astronaut() (Lena测试图)")
print(f"原图尺寸: {img.shape}")

# 确保尺寸为16的整数倍（方便16x16分块）
h, w = img.shape
h = h - (h % 16)
w = w - (w % 16)
img = img[:h, :w]
print(f"裁剪后尺寸(16的倍数): {h} x {w}")
print(f"下采样后尺寸(1/4): {h//2} x {w//2}")

# 显示原图
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title("Original (Astronaut/Lena)")
plt.axis('off')
plt.show()

# ----------------------------- 2. 定义下采样和恢复函数 -----------------------------
def block_bilinear_downsample(img, block_size=16, target_block_size=8):
    """
    将图像分块，每块通过双线性插值缩小到 target_block_size x target_block_size
    然后组合成完整缩小图像
    block_size: 原始块大小（16x16）
    target_block_size: 目标块大小（8x8）
    缩小倍数 = block_size / target_block_size = 2（即原图1/4面积）
    """
    h, w = img.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    # 计算缩小后的图像尺寸
    h_small = h_blocks * target_block_size
    w_small = w_blocks * target_block_size
    
    small_img = np.zeros((h_small, w_small), dtype=np.uint8)
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            # 提取当前16x16块
            block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            # 使用双线性插值将16x16块缩小到8x8
            block_small = cv2.resize(block, (target_block_size, target_block_size), 
                                     interpolation=cv2.INTER_LINEAR)
            # 放入结果图像
            small_img[i*target_block_size:(i+1)*target_block_size, 
                      j*target_block_size:(j+1)*target_block_size] = block_small
    
    return small_img

def block_bilinear_downsample_with_gaussian(img, block_size=16, target_block_size=8, sigma=1.0):
    """
    先高斯预滤波，再进行分块双线性插值下采样
    """
    # 高斯预滤波
    ksize = int(2 * np.ceil(3*sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    img_filtered = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    # 分块双线性下采样
    return block_bilinear_downsample(img_filtered, block_size, target_block_size)

def restore_image(small_img, original_shape, method='bilinear'):
    """将缩小图像恢复到原始尺寸"""
    h_orig, w_orig = original_shape
    if method == 'nearest':
        return cv2.resize(small_img, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        return cv2.resize(small_img, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    elif method == 'bicubic':
        return cv2.resize(small_img, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

def mse(img1, img2):
    """计算均方误差"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def psnr(img1, img2):
    """计算峰值信噪比"""
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse_val)

def show_spectrum(img, title, ax):
    """显示频谱图（中心化+对数）"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    ax.imshow(magnitude, cmap='gray')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def dct2(block):
    """二维DCT变换"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def low_freq_energy_ratio(img, block_size=8, low_ratio=0.5):
    """计算分块DCT的低频能量比例"""
    h, w = img.shape
    h_crop = h - h % block_size
    w_crop = w - w % block_size
    img_crop = img[:h_crop, :w_crop].astype(float)
    
    total_energy = 0.0
    low_energy = 0.0
    low_size = int(block_size * low_ratio)
    
    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = img_crop[i:i+block_size, j:j+block_size]
            dct_block = dct2(block)
            energy = np.sum(dct_block ** 2)
            total_energy += energy
            low_energy += np.sum(dct_block[:low_size, :low_size] ** 2)
    
    return low_energy / total_energy if total_energy > 0 else 0

def show_dct_coeffs(img, title, ax):
    """显示整图DCT系数（对数显示）"""
    img_float = img.astype(float)
    dct_coeffs = dct2(img_float)
    dct_log = np.log(np.abs(dct_coeffs) + 1)
    ax.imshow(dct_log, cmap='gray')
    ax.set_title(title, fontsize=10)
    ax.axis('off')

# ----------------------------- 3. 下采样（缩小为原来的1/4） -----------------------------

# 方法1: 直接分块双线性下采样（无预滤波）
small_direct = block_bilinear_downsample(img, block_size=16, target_block_size=8)

# 方法2: 高斯预滤波 + 分块双线性下采样
sigma = 1.0  # 对于16x16块下采样到8x8，sigma=1.0合适
small_gaussian = block_bilinear_downsample_with_gaussian(img, block_size=16, target_block_size=8, sigma=sigma)

print(f"原图尺寸: {img.shape}")
print(f"下采样后尺寸: {small_direct.shape}")
print(f"理论缩小比例: {img.shape[0]//small_direct.shape[0]}x{img.shape[1]//small_direct.shape[1]} = 2x2倍（面积1/4）")
print(f"高斯滤波核sigma: {sigma}")

# 显示下采样结果对比
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title(f"Original\n{img.shape}")
plt.subplot(132), plt.imshow(small_direct, cmap='gray')
plt.title(f"Direct block-wise \nbilinear downsampling\n{small_direct.shape}")
plt.subplot(133), plt.imshow(small_gaussian, cmap='gray')
plt.title(f"Gaussian pre-filtering + \nblock-wise bilinear downsampling\n{small_gaussian.shape}")
plt.tight_layout()
plt.show()

# ----------------------------- 4. 图像恢复（对两种下采样结果分别用三种内插方法） -----------------------------
print("\n" + "="*70)
print("图像恢复（使用三种内插方法恢复到原始尺寸）")
print("="*70)

# 对直接分块双线性下采样的结果进行恢复
restore_direct_nn = restore_image(small_direct, img.shape, 'nearest')
restore_direct_bilinear = restore_image(small_direct, img.shape, 'bilinear')
restore_direct_bicubic = restore_image(small_direct, img.shape, 'bicubic')

# 对高斯预滤波+分块双线性下采样的结果进行恢复
restore_gaussian_nn = restore_image(small_gaussian, img.shape, 'nearest')
restore_gaussian_bilinear = restore_image(small_gaussian, img.shape, 'bilinear')
restore_gaussian_bicubic = restore_image(small_gaussian, img.shape, 'bicubic')

print("所有恢复完成")

# ----------------------------- 5. 空间域比较（MSE和PSNR） -----------------------------
print("\n" + "="*70)
print("空间域比较（MSE和PSNR）")
print("="*70)

print("\n【情况1：直接分块双线性下采样后恢复】")
print("-" * 70)
print(f"{'内插方法':<12} {'MSE':<15} {'PSNR (dB)':<15}")
print("-" * 70)
print(f"{'最近邻':<12} {mse(img, restore_direct_nn):<15.4f} {psnr(img, restore_direct_nn):<15.2f}")
print(f"{'双线性':<12} {mse(img, restore_direct_bilinear):<15.4f} {psnr(img, restore_direct_bilinear):<15.2f}")
print(f"{'双三次':<12} {mse(img, restore_direct_bicubic):<15.4f} {psnr(img, restore_direct_bicubic):<15.2f}")
print("-" * 70)

print("\n【情况2：高斯预滤波 + 分块双线性下采样后恢复】")
print("-" * 70)
print(f"{'内插方法':<12} {'MSE':<15} {'PSNR (dB)':<15}")
print("-" * 70)
print(f"{'最近邻':<12} {mse(img, restore_gaussian_nn):<15.4f} {psnr(img, restore_gaussian_nn):<15.2f}")
print(f"{'双线性':<12} {mse(img, restore_gaussian_bilinear):<15.4f} {psnr(img, restore_gaussian_bilinear):<15.2f}")
print(f"{'双三次':<12} {mse(img, restore_gaussian_bicubic):<15.4f} {psnr(img, restore_gaussian_bicubic):<15.2f}")
print("-" * 70)

# 显示所有恢复图像对比
fig, axes = plt.subplots(2, 4, figsize=(16, 9))

# 第一行：直接下采样恢复
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original", fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(restore_direct_nn, cmap='gray')
axes[0, 1].set_title(f"Direct downsampling\n + Nearest Neighbor\nPSNR: {psnr(img, restore_direct_nn):.2f} dB", fontsize=10)
axes[0, 1].axis('off')

axes[0, 2].imshow(restore_direct_bilinear, cmap='gray')
axes[0, 2].set_title(f"Direct downsampling\n + Bilinear\nPSNR: {psnr(img, restore_direct_bilinear):.2f} dB", fontsize=10)
axes[0, 2].axis('off')

axes[0, 3].imshow(restore_direct_bicubic, cmap='gray')
axes[0, 3].set_title(f"Direct downsampling\n + Bicubic\nPSNR: {psnr(img, restore_direct_bicubic):.2f} dB", fontsize=10)
axes[0, 3].axis('off')

# 第二行：高斯预滤波下采样恢复
axes[1, 0].imshow(small_gaussian, cmap='gray')
axes[1, 0].set_title(f"Gaussian pre-filtering + \nblock-wise bilinear downsampling\n{small_gaussian.shape}", fontsize=10)
axes[1, 0].axis('off')

axes[1, 1].imshow(restore_gaussian_nn, cmap='gray')
axes[1, 1].set_title(f"Gaussian pre-filtering + \nNearest Neighbor\nPSNR: {psnr(img, restore_gaussian_nn):.2f} dB", fontsize=10)
axes[1, 1].axis('off')

axes[1, 2].imshow(restore_gaussian_bilinear, cmap='gray')
axes[1, 2].set_title(f"Gaussian pre-filtering\n + Bilinear\nPSNR: {psnr(img, restore_gaussian_bilinear):.2f} dB", fontsize=10)
axes[1, 2].axis('off')

axes[1, 3].imshow(restore_gaussian_bicubic, cmap='gray')
axes[1, 3].set_title(f"Gaussian pre-filtering\n + Bicubic\nPSNR: {psnr(img, restore_gaussian_bicubic):.2f} dB", fontsize=10)
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

# ----------------------------- 6. 傅里叶变换分析 -----------------------------
print("\n" + "="*70)
print("傅里叶变换频谱分析")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

# 第一行：直接下采样相关频谱
show_spectrum(img, "Original image spectrum", axes[0, 0])
show_spectrum(small_direct, "Spectrum of direct block-wise\nbilinear downsampling", axes[0, 1])
show_spectrum(restore_direct_nn, "Direct + Nearest Neighbor\nRecovered spectrum", axes[0, 2])
show_spectrum(restore_direct_bilinear, "Direct + Bilinear\nRecovered Spectrum", axes[0, 3])

# 第二行：高斯预滤波下采样相关频谱
show_spectrum(small_gaussian, "Gaussian Pre-filtering + \nBlock-wise Bilinear Downsampling Spectrum", axes[1, 0])
show_spectrum(restore_gaussian_nn, "Gaussian + Nearest Neighbor\nRecovered Spectrum", axes[1, 1])
show_spectrum(restore_gaussian_bilinear, "Gaussian + Bilinear\nRecovered Spectrum", axes[1, 2])
show_spectrum(restore_gaussian_bicubic, "Gaussian + Bicubic\nRecovered Spectrum", axes[1, 3])

plt.tight_layout()
plt.show()

# ----------------------------- 7. DCT 分析 -----------------------------
print("\n" + "="*70)
print("DCT 低频能量比例分析（8x8分块，低频区域4x4）")
print("="*70)

# 计算各种恢复图像的低频能量比例
ratio_orig = low_freq_energy_ratio(img)

ratio_direct_nn = low_freq_energy_ratio(restore_direct_nn)
ratio_direct_bilinear = low_freq_energy_ratio(restore_direct_bilinear)
ratio_direct_bicubic = low_freq_energy_ratio(restore_direct_bicubic)

ratio_gaussian_nn = low_freq_energy_ratio(restore_gaussian_nn)
ratio_gaussian_bilinear = low_freq_energy_ratio(restore_gaussian_bilinear)
ratio_gaussian_bicubic = low_freq_energy_ratio(restore_gaussian_bicubic)

print("\n【情况1：直接分块双线性下采样后恢复】")
print("-" * 70)
print(f"{'内插方法':<12} {'低频能量比例':<20}")
print("-" * 70)
print(f"{'原图':<12} {ratio_orig:<20.6f}")
print(f"{'最近邻':<12} {ratio_direct_nn:<20.6f}")
print(f"{'双线性':<12} {ratio_direct_bilinear:<20.6f}")
print(f"{'双三次':<12} {ratio_direct_bicubic:<20.6f}")
print("-" * 70)

print("\n【情况2：高斯预滤波 + 分块双线性下采样后恢复】")
print("-" * 70)
print(f"{'内插方法':<12} {'低频能量比例':<20}")
print("-" * 70)
print(f"{'原图':<12} {ratio_orig:<20.6f}")
print(f"{'最近邻':<12} {ratio_gaussian_nn:<20.6f}")
print(f"{'双线性':<12} {ratio_gaussian_bilinear:<20.6f}")
print(f"{'双三次':<12} {ratio_gaussian_bicubic:<20.6f}")
print("-" * 70)

print("\n【低频能量比例对比分析】")
print(f"原图低频比例: {ratio_orig:.6f}")
print(f"直接下采样恢复平均低频比例: {(ratio_direct_nn + ratio_direct_bilinear + ratio_direct_bicubic)/3:.6f}")
print(f"高斯预滤波恢复平均低频比例: {(ratio_gaussian_nn + ratio_gaussian_bilinear + ratio_gaussian_bicubic)/3:.6f}")

# 显示DCT系数图
fig, axes = plt.subplots(2, 4, figsize=(16, 9))

# 第一行：直接下采样相关DCT
show_dct_coeffs(img, "Original image DCT\ncoefficients", axes[0, 0])
show_dct_coeffs(small_direct, "Direct block-wise bilinearly\n downscaled image DCT", axes[0, 1])
show_dct_coeffs(restore_direct_nn, "Direct + Nearest Neighbor\n – Recovered DCT", axes[0, 2])
show_dct_coeffs(restore_direct_bilinear, "Direct + Bilinear \n– Recovered DCT", axes[0, 3])

# 第二行：高斯预滤波相关DCT
show_dct_coeffs(small_gaussian, "Gaussian pre-filtered + \nblock-wise bilinearly downscaled image DCT", axes[1, 0])
show_dct_coeffs(restore_gaussian_nn, "Gaussian + Nearest Neighbor\n – Recovered DCT", axes[1, 1])
show_dct_coeffs(restore_gaussian_bilinear, "Gaussian + Bilinear\n Recovery DCT", axes[1, 2])
show_dct_coeffs(restore_gaussian_bicubic, "Gaussian + Bicubic\n Recovery DCT", axes[1, 3])

plt.tight_layout()
plt.show()
