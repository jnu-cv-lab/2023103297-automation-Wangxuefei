import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import data, color, io
import os

class FrequencyAnalyzer:
    def __init__(self, image):
        """初始化：确保图像为灰度图"""
        self.image = image if len(image.shape) == 2 else color.rgb2gray(image)
        self.name = 'camera'
        self.block_size = 16
        os.makedirs('output', exist_ok=True)
    
    def spatial_gradient_method(self):
        """
        空域梯度方法：估计最高频率
        公式: f_max² = E[|∇I|²] / (4π² Var(I))
        """
        h, w = self.image.shape
        n_h, n_w = h // self.block_size, w // self.block_size
        freqs = np.zeros((n_h, n_w))
        
        # Sobel算子
        sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        
        for i in range(n_h):
            for j in range(n_w):
                block = self.image[i*self.block_size:(i+1)*self.block_size, 
                                  j*self.block_size:(j+1)*self.block_size]
                grad_x = convolve(block, sobel_x)
                grad_y = convolve(block, sobel_y)
                mean_grad_sq = np.mean(grad_x**2 + grad_y**2)
                var_i = np.var(block)
                freqs[i, j] = np.sqrt(mean_grad_sq / (4 * np.pi**2 * (var_i + 1e-10)))
        
        return freqs
    
    def fft_rms_method(self):
        """
        FFT RMS方法：基于功率谱的二阶矩
        公式: f_rms = sqrt(∑(f² × P(f)) / ∑P(f))
        """
        h, w = self.image.shape
        n_h, n_w = h // self.block_size, w // self.block_size
        freqs = np.zeros((n_h, n_w))
        
        for i in range(n_h):
            for j in range(n_w):
                block = self.image[i*self.block_size:(i+1)*self.block_size, 
                                  j*self.block_size:(j+1)*self.block_size]
                
                # 应用汉宁窗
                window = np.hanning(self.block_size)[:, None] * np.hanning(self.block_size)[None, :]
                block_windowed = block * window
                
                # FFT和功率谱
                fft = np.fft.fft2(block_windowed)
                fft_shifted = np.fft.fftshift(fft)
                power = np.abs(fft_shifted)**2
                
                # 频率坐标
                fx, fy = np.meshgrid(np.fft.fftfreq(self.block_size), 
                                    np.fft.fftfreq(self.block_size))
                radial_freq = np.sqrt(fx**2 + fy**2)
                
                # RMS频率
                if np.sum(power) > 0:
                    freqs[i, j] = np.sqrt(np.sum(radial_freq**2 * power) / np.sum(power))
        
        return freqs
    
    def compare_gradient_vs_fft_rms(self):
        """比较空域梯度法和FFT RMS法的一致性"""
        print("\n" + "="*60)
        print("比较：空域梯度法 vs FFT RMS法（功率谱二阶矩）")
        print("="*60)
        
        # 计算两种方法的频率
        gradient_freq = self.spatial_gradient_method()
        fft_rms_freq = self.fft_rms_method()
        
        # 打印统计信息
        print(f"\n块大小: {self.block_size}x{self.block_size}")
        print(f"图像分块数: {gradient_freq.shape[0]} x {gradient_freq.shape[1]}")
        print(f"\n频率范围:")
        print(f"  空域梯度法: [{gradient_freq.min():.6f}, {gradient_freq.max():.6f}]")
        print(f"  FFT RMS法: [{fft_rms_freq.min():.6f}, {fft_rms_freq.max():.6f}]")
        print(f"  频率均值: 梯度={gradient_freq.mean():.6f}, FFT RMS={fft_rms_freq.mean():.6f}")
        
        # 保存频率图为numpy数组（调试用）
        np.save(f'output/{self.name}_gradient_freq.npy', gradient_freq)
        np.save(f'output/{self.name}_fft_rms_freq.npy', fft_rms_freq)
        
        # 保存为图像（使用matplotlib保存，确保可见）
        plt.figure(figsize=(10, 4))
        plt.subplot(1,2,1)
        plt.imshow(gradient_freq, cmap='hot')
        plt.colorbar()
        plt.title('Spatial Gradient Method')
        plt.axis('off')
        plt.savefig(f'output/{self.name}_gradient.png', dpi=150, bbox_inches='tight')
        
        plt.subplot(1,2,2)
        plt.imshow(fft_rms_freq, cmap='hot')
        plt.colorbar()
        plt.title('FFT RMS Method')
        plt.axis('off')
        plt.savefig(f'output/{self.name}_fft_rms.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 计算一致性指标
        valid = ~(np.isnan(gradient_freq) | np.isnan(fft_rms_freq))
        correlation = np.corrcoef(gradient_freq[valid], fft_rms_freq[valid])[0, 1]
        mae = np.mean(np.abs(gradient_freq[valid] - fft_rms_freq[valid]))
        rmse = np.sqrt(np.mean((gradient_freq[valid] - fft_rms_freq[valid])**2))
        
        print(f"\n一致性指标:")
        print(f"  相关系数: {correlation:.6f}")
        print(f"  平均绝对误差 (MAE): {mae:.6f}")
        print(f"  均方根误差 (RMSE): {rmse:.6f}")
        
        # 创建综合对比图
        fig = plt.figure(figsize=(16, 10))
        
        # 原始图像
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(self.image, cmap='gray')
        ax1.set_title(f'Original Image\n{self.image.shape[0]}x{self.image.shape[1]}')
        ax1.axis('off')
        
        # 空域梯度法
        ax2 = plt.subplot(2, 3, 2)
        im1 = ax2.imshow(gradient_freq, cmap='hot')
        ax2.set_title(f'Spatial Gradient Method\nMean={gradient_freq.mean():.3f}')
        ax2.axis('off')
        plt.colorbar(im1, ax=ax2, fraction=0.046)
        
        # FFT RMS法
        ax3 = plt.subplot(2, 3, 3)
        im2 = ax3.imshow(fft_rms_freq, cmap='hot')
        ax3.set_title(f'FFT RMS Method\nMean={fft_rms_freq.mean():.3f}')
        ax3.axis('off')
        plt.colorbar(im2, ax=ax3, fraction=0.046)
        
        # 散点图
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(gradient_freq[valid], fft_rms_freq[valid], s=20, alpha=0.5, c='blue')
        max_val = max(gradient_freq.max(), fft_rms_freq.max())
        ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
        ax4.set_xlabel('Spatial Gradient Frequency')
        ax4.set_ylabel('FFT RMS Frequency')
        ax4.set_title(f'Correlation: {correlation:.4f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 误差分布
        ax5 = plt.subplot(2, 3, 5)
        errors = gradient_freq[valid] - fft_rms_freq[valid]
        ax5.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.axvline(x=np.mean(errors), color='b', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
        ax5.set_xlabel('Frequency Error (Gradient - FFT)')
        ax5.set_ylabel('Count')
        ax5.set_title(f'Error Distribution\nMAE: {mae:.4f}, RMSE: {rmse:.4f}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 频率分布对比
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(gradient_freq[valid], bins=30, alpha=0.5, label='Spatial Gradient', edgecolor='black')
        ax6.hist(fft_rms_freq[valid], bins=30, alpha=0.5, label='FFT RMS', edgecolor='black')
        ax6.set_xlabel('Frequency')
        ax6.set_ylabel('Count')
        ax6.set_title('Frequency Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Comparison: Spatial Gradient vs FFT RMS Method\nBlock Size: {self.block_size}x{self.block_size}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'output/{self.name}_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("="*60)
        

# 运行测试
if __name__ == "__main__":
    # 使用scikit-image内置图像
    img = data.camera()
    print(f"图像尺寸: {img.shape}")
    
    # 创建分析器并比较
    analyzer = FrequencyAnalyzer(img)
    results = analyzer.compare_gradient_vs_fft_rms()
    
    print("\n" + "="*60)
    print("结果已保存到 'output' 文件夹")
    print("生成的文件:")
    print("  - camera_gradient.png: 空域梯度法频率图")
    print("  - camera_fft_rms.png: FFT RMS法频率图")
    print("  - camera_comparison.png: 综合对比图")
    print("  - camera_gradient_freq.npy: 梯度法数据")
    print("  - camera_fft_rms_freq.npy: FFT法数据")
    print("="*60)
    