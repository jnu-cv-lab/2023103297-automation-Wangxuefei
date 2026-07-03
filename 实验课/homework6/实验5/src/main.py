import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from skimage import data
import os

# 创建 output 文件夹
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==================== 1. 加载图像并取第200行像素 ====================
img = data.camera()  # 512x512 灰度图
row_idx = 200
N = 16

x_original = img[row_idx, :N].astype(float)

print("=" * 60)
print(f"原始图像第{row_idx}行像素 (前16个):")
print(x_original.astype(int))
print(f"均值: {np.mean(x_original):.2f}, 方差: {np.var(x_original):.2f}")
print("=" * 60)

# ==================== 2. 延拓方式对比 ====================
periodic_extension = np.tile(x_original, 2)
mirrored = x_original[::-1]
dct_extension = np.concatenate([mirrored, x_original])

print("\n【DFT 周期延拓】边界跳变:", int(x_original[-1]), "→", int(x_original[0]))
print("【DCT 偶对称延拓】边界连续:", int(mirrored[-1]), "=", int(x_original[0]))

# ==================== 3. 可视化延拓对比 ====================
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

n_periodic = np.arange(2*N)
axes[0].stem(n_periodic, periodic_extension, basefmt=" ", linefmt='b-', markerfmt='bo')
axes[0].axvline(x=N-0.5, color='r', linestyle='--', linewidth=2, label='Periodic boundary')
axes[0].set_title(f'DFT Periodic Extension (Row {row_idx})', fontsize=12)
axes[0].set_xlabel('Sample index n')
axes[0].set_ylabel('Gray value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

n_dct = np.arange(2*N)
axes[1].stem(n_dct, dct_extension, basefmt=" ", linefmt='g-', markerfmt='go')
axes[1].axvline(x=N-0.5, color='r', linestyle='--', linewidth=2, label='Symmetry axis (n = -0.5)')
axes[1].set_title(f'DCT Even Symmetric Extension (Row {row_idx})', fontsize=12)
axes[1].set_xlabel('Sample index n')
axes[1].set_ylabel('Gray value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'extension_comparison_row200.png'), dpi=150, bbox_inches='tight')
plt.show()

# ==================== 4. 计算 DFT 和 DCT 系数 ====================
# DFT (复数)
X_dft = np.fft.fft(x_original)
X_dft_mag = np.abs(X_dft)

# DCT (Type-II)
X_dct = dct(x_original, type=2, norm='ortho')

print("\n" + "=" * 60)
print("变换系数:")
print("=" * 60)
print("\nDFT 幅度谱:")
for i in range(N):
    print(f"  k={i:2d}: {X_dft_mag[i]:8.2f}")
    
print("\nDCT 系数:")
for i in range(N):
    print(f"  k={i:2d}: {X_dct[i]:8.2f}")

# ==================== 5. 能量集中性对比 ====================
def energy_ratio(coeff, top_k):
    """计算前 top_k 个系数占总能量的比例"""
    total_energy = np.sum(coeff ** 2)
    if total_energy == 0:
        return 0
    top_energy = np.sum(coeff[:top_k] ** 2)
    return top_energy / total_energy

print("\n" + "=" * 60)
print("能量集中性分析:")
print("=" * 60)

print("\n前3个系数能量占比:")
dft_ratio_3 = energy_ratio(X_dft_mag, 3)
dct_ratio_3 = energy_ratio(X_dct, 3)
print(f"  DFT: {dft_ratio_3:.4f} ({dft_ratio_3*100:.2f}%)")
print(f"  DCT: {dct_ratio_3:.4f} ({dct_ratio_3*100:.2f}%)")

print("\n前5个系数能量占比:")
dft_ratio_5 = energy_ratio(X_dft_mag, 5)
dct_ratio_5 = energy_ratio(X_dct, 5)
print(f"  DFT: {dft_ratio_5:.4f} ({dft_ratio_5*100:.2f}%)")
print(f"  DCT: {dct_ratio_5:.4f} ({dct_ratio_5*100:.2f}%)")

print("\n前8个系数能量占比:")
dft_ratio_8 = energy_ratio(X_dft_mag, 8)
dct_ratio_8 = energy_ratio(X_dct, 8)
print(f"  DFT: {dft_ratio_8:.4f} ({dft_ratio_8*100:.2f}%)")
print(f"  DCT: {dct_ratio_8:.4f} ({dct_ratio_8*100:.2f}%)")

# ==================== 6. 可视化频谱对比 ====================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

k = np.arange(N)

axes[0].stem(k, X_dft_mag, basefmt=" ", linefmt='b-', markerfmt='bo')
axes[0].set_title(f'DFT Magnitude Spectrum (Row {row_idx})', fontsize=12)
axes[0].set_xlabel('Frequency index k')
axes[0].set_ylabel('Magnitude')
axes[0].grid(True, alpha=0.3)

axes[1].stem(k, X_dct, basefmt=" ", linefmt='r-', markerfmt='ro')
axes[1].set_title(f'DCT Coefficients (Row {row_idx}, Type-II)', fontsize=12)
axes[1].set_xlabel('Frequency index k')
axes[1].set_ylabel('Coefficient value')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spectrum_comparison_row200.png'), dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. 显示原始信号波形 ====================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x_original, 'b-o', linewidth=2, markersize=8)
ax.set_title(f'Original Signal (Row {row_idx}, First {N} pixels)', fontsize=12)
ax.set_xlabel('Pixel index n')
ax.set_ylabel('Gray value')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'original_signal.png'), dpi=150, bbox_inches='tight')
plt.show()

# ==================== 8. 打印详细分析 ====================
print("\n" + "=" * 60)
print("详细分析:")
print("=" * 60)
print(f"\n原始信号首尾值: {int(x_original[0])} vs {int(x_original[-1])}")
print(f"首尾差值: {abs(int(x_original[0]) - int(x_original[-1]))}")

# 计算边界跳变的影响
boundary_jump = abs(x_original[-1] - x_original[0])
print(f"\n边界跳变幅度: {boundary_jump}")

# ==================== 9. 显示原始图像 ====================
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, cmap='gray')
ax.axhline(y=row_idx, color='r', linestyle='--', linewidth=2, label=f'Selected row {row_idx}')
ax.set_title('Camera Image with Selected Row', fontsize=14)
ax.set_xlabel('Column index')
ax.set_ylabel('Row index')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'selected_row.png'), dpi=150, bbox_inches='tight')
plt.show()

# ==================== 10. 打印延拓序列详细列表 ====================
print("\n" + "=" * 60)
print("延拓序列详细列表:")
print("=" * 60)

print("\n【DFT 周期延拓序列】 (索引 0~31):")
print("Index: ", end="")
for i in range(2*N):
    print(f"{i:3d}", end=" ")
print()
print("Value: ", end="")
for val in periodic_extension:
    print(f"{int(val):3d}", end=" ")
print()

print("\n【DCT 偶对称延拓序列】 (索引 0~31):")
print("Index: ", end="")
for i in range(2*N):
    print(f"{i:3d}", end=" ")
print()
print("Value: ", end="")
for val in dct_extension:
    print(f"{int(val):3d}", end=" ")
print()

print("\n" + "=" * 60)
print(f"所有图片已保存到 '{output_dir}' 文件夹中:")
print(f"  - extension_comparison_row200.png")
print(f"  - spectrum_comparison_row200.png")
print(f"  - original_signal.png")
print(f"  - selected_row.png")
print("=" * 60)