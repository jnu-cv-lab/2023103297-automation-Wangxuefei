import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10, sqrt

def calculate_psnr(img1, img2):
    """
    计算两幅图像的PSNR（峰值信噪比）
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def downsample_channel(channel, factor):
    """
    对通道进行下采样
    factor: 下采样因子，例如2表示尺寸减半
    """
    height, width = channel.shape
    new_height = height // factor
    new_width = width // factor
    
    # 使用最近邻下采样（也可以使用其他方法）
    downsampled = cv2.resize(channel, (new_width, new_height), 
                             interpolation=cv2.INTER_NEAREST)
    return downsampled

def upsample_channel(channel, original_shape, method='bilinear'):
    """
    对通道进行上采样恢复到原始尺寸
    method: 插值方法 'nearest', 'bilinear', 'bicubic'
    """
    height, width = original_shape
    
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    
    upsampled = cv2.resize(channel, (width, height), interpolation=interpolation)
    return upsampled

def reconstruct_image(y_channel, cb_channel, cr_channel):
    """
    使用Y、Cb、Cr通道重建图像
    """
    # 合并三个通道（注意OpenCV使用YCrCb顺序）
    ycrcb_reconstructed = cv2.merge([y_channel, cr_channel, cb_channel])
    # 转换回BGR
    bgr_reconstructed = cv2.cvtColor(ycrcb_reconstructed, cv2.COLOR_YCrCb2BGR)
    return bgr_reconstructed

def main():
    # 1. 读入彩色图像
    image_path = '/home/wxf81/cv-course/实验22/images/input/test.jpg' 
    original_bgr = cv2.imread(image_path)
    
    if original_bgr is None:
        print("错误：无法读取图像，请检查路径")
        return
    
    print(f"原始图像尺寸: {original_bgr.shape}")
    
    # 2. 转换到YCbCr色彩空间
    original_ycrcb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
    
    # 提取三个通道
    y_channel = original_ycrcb[:, :, 0].copy()
    cr_channel = original_ycrcb[:, :, 1].copy()
    cb_channel = original_ycrcb[:, :, 2].copy()
    
    print(f"原始Y通道范围: [{y_channel.min()}, {y_channel.max()}]")
    print(f"原始Cb通道范围: [{cb_channel.min()}, {cb_channel.max()}]")
    print(f"原始Cr通道范围: [{cr_channel.min()}, {cr_channel.max()}]")
    
    # 定义下采样因子
    downsample_factors = [2, 4, 8]  # 分别进行2倍、4倍、8倍下采样
    interpolation_methods = ['nearest', 'bilinear', 'bicubic']
    
    # 存储结果
    results = {}
    psnr_values = {}
    
    # 创建输出目录
    output_dir = '/home/wxf81/cv-course/实验22/images/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 对不同下采样因子和插值方法进行实验
    for factor in downsample_factors:
        print(f"\n{'='*50}")
        print(f"下采样因子: {factor}x")
        print(f"{'='*50}")
        
        results[factor] = {}
        psnr_values[factor] = {}
        
        # 对Cb和Cr通道进行下采样
        cb_downsampled = downsample_channel(cb_channel, factor)
        cr_downsampled = downsample_channel(cr_channel, factor)
        
        # 保存下采样后的Cb和Cr通道
        cb_down_path = os.path.join(output_dir, f'cb_downsampled_{factor}x.jpg')
        cr_down_path = os.path.join(output_dir, f'cr_downsampled_{factor}x.jpg')
        cv2.imwrite(cb_down_path, cb_downsampled)
        cv2.imwrite(cr_down_path, cr_downsampled)
        
        print(f"下采样后Cb通道尺寸: {cb_downsampled.shape} -> 已保存: {os.path.basename(cb_down_path)}")
        print(f"下采样后Cr通道尺寸: {cr_downsampled.shape} -> 已保存: {os.path.basename(cr_down_path)}")
        
        for method in interpolation_methods:
            # 上采样恢复Cb和Cr通道到原始尺寸
            cb_upsampled = upsample_channel(cb_downsampled, cb_channel.shape, method)
            cr_upsampled = upsample_channel(cr_downsampled, cr_channel.shape, method)
            
            # 重建图像（使用原始Y通道和恢复的Cb、Cr通道）
            reconstructed_bgr = reconstruct_image(y_channel, cb_upsampled, cr_upsampled)
            
            # 计算PSNR
            psnr = calculate_psnr(original_bgr, reconstructed_bgr)
            psnr_values[factor][method] = psnr
            
            # 保存结果
            results[factor][method] = reconstructed_bgr
            
            # 保存重建图像
            recon_path = os.path.join(output_dir, f'reconstructed_{factor}x_{method}.jpg')
            cv2.imwrite(recon_path, reconstructed_bgr)
            
            print(f"插值方法: {method:10s} | PSNR: {psnr:.2f} dB -> 已保存: {os.path.basename(recon_path)}")
    
    # 4. 保存原始通道图像
    cv2.imwrite(os.path.join(output_dir, 'original_y_channel.jpg'), y_channel)
    cv2.imwrite(os.path.join(output_dir, 'original_cb_channel.jpg'), cb_channel)
    cv2.imwrite(os.path.join(output_dir, 'original_cr_channel.jpg'), cr_channel)
    cv2.imwrite(os.path.join(output_dir, 'original_rgb.jpg'), original_bgr)
    
    print("\n原始通道图像已保存")
    
    # 5. 显示和保存结果对比图
    plt.figure(figsize=(20, 15))
    
    # 显示原始图像
    plt.subplot(len(downsample_factors) + 1, len(interpolation_methods) + 1, 1)
    plt.imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示各通道（原始）
    plt.subplot(len(downsample_factors) + 1, len(interpolation_methods) + 1, 2)
    plt.imshow(y_channel, cmap='gray')
    plt.title('Y Channel')
    plt.axis('off')
    
    plt.subplot(len(downsample_factors) + 1, len(interpolation_methods) + 1, 3)
    plt.imshow(cb_channel, cmap='gray')
    plt.title('Cb Channel')
    plt.axis('off')
    
    plt.subplot(len(downsample_factors) + 1, len(interpolation_methods) + 1, 4)
    plt.imshow(cr_channel, cmap='gray')
    plt.title('Cr Channel')
    plt.axis('off')
    
    # 显示不同下采样因子和插值方法的重建结果
    plot_idx = 5  # 从第5个子图开始
    for i, factor in enumerate(downsample_factors):
        for j, method in enumerate(interpolation_methods):
            reconstructed = results[factor][method]
            psnr = psnr_values[factor][method]
            
            plt.subplot(len(downsample_factors) + 1, len(interpolation_methods) + 1, plot_idx)
            plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
            plt.title(f'{factor}x downsample\n{method} interpolation\nPSNR: {psnr:.2f} dB')
            plt.axis('off')
            plot_idx += 1
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'all_results_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {comparison_path}")
    plt.show()
    
    # 6. 绘制PSNR对比图
    plt.figure(figsize=(12, 8))
    
    for method in interpolation_methods:
        psnr_list = [psnr_values[factor][method] for factor in downsample_factors]
        plt.plot(downsample_factors, psnr_list, marker='o', linewidth=2, 
                markersize=8, label=method.capitalize())
    
    plt.xlabel('Downsampling Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR Comparison for Different Downsampling Factors and Interpolation Methods', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(downsample_factors)
    
    # 添加参考线
    plt.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Good Quality (30 dB)')
    plt.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Acceptable (25 dB)')
    plt.legend(fontsize=10)
    
    psnr_plot_path = os.path.join(output_dir, 'psnr_comparison.png')
    plt.savefig(psnr_plot_path, dpi=300, bbox_inches='tight')
    print(f"PSNR对比图已保存: {psnr_plot_path}")
    plt.show()
    
    # 7. 保存详细的PSNR表格
    with open(os.path.join(output_dir, 'psnr_results.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PSNR Results (dB)\n")
        f.write("="*70 + "\n")
        f.write(f"{'Factor':<12}")
        for method in interpolation_methods:
            f.write(f"{method.capitalize():<18}")
        f.write("\n")
        f.write("-"*70 + "\n")
        
        for factor in downsample_factors:
            f.write(f"{factor}x{'':<8}")
            for method in interpolation_methods:
                f.write(f"{psnr_values[factor][method]:<18.2f}")
            f.write("\n")
        f.write("-"*70 + "\n")
        
        # 添加平均值
        f.write(f"{'Average':<12}")
        for method in interpolation_methods:
            avg_psnr = np.mean([psnr_values[factor][method] for factor in downsample_factors])
            f.write(f"{avg_psnr:<18.2f}")
        f.write("\n")
    
    print(f"PSNR结果表格已保存: {os.path.join(output_dir, 'psnr_results.txt')}")
    
    # 8. 保存分析报告
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("YCbCr色彩空间下采样与插值重建实验分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("一、实验参数设置\n")
        f.write("-"*50 + "\n")
        f.write(f"原始图像尺寸: {original_bgr.shape}\n")
        f.write(f"下采样因子: {downsample_factors}\n")
        f.write(f"插值方法: {interpolation_methods}\n\n")
        
        f.write("二、PSNR结果分析\n")
        f.write("-"*50 + "\n")
        for factor in downsample_factors:
            f.write(f"\n{factor}x下采样:\n")
            for method in interpolation_methods:
                f.write(f"  {method.capitalize()}: {psnr_values[factor][method]:.2f} dB\n")
        
        f.write("\n三、下采样因子对图像质量的影响:\n")
        f.write("-"*50 + "\n")
        for factor in downsample_factors:
            avg_psnr = np.mean([psnr_values[factor][method] for method in interpolation_methods])
            f.write(f"  {factor}x下采样: 平均PSNR = {avg_psnr:.2f} dB")
            if avg_psnr > 30:
                f.write(" (质量较好)\n")
            elif avg_psnr > 25:
                f.write(" (质量中等)\n")
            else:
                f.write(" (质量较差)\n")
        
        f.write("\n四、插值方法对图像质量的影响:\n")
        f.write("-"*50 + "\n")
        for method in interpolation_methods:
            avg_psnr = np.mean([psnr_values[factor][method] for factor in downsample_factors])
            f.write(f"  {method.capitalize()}: 平均PSNR = {avg_psnr:.2f} dB\n")
        
        f.write("\n五、最佳重建效果:\n")
        f.write("-"*50 + "\n")
        best_psnr = -float('inf')
        best_config = None
        for factor in downsample_factors:
            for method in interpolation_methods:
                if psnr_values[factor][method] > best_psnr:
                    best_psnr = psnr_values[factor][method]
                    best_config = (factor, method)
        f.write(f"  下采样因子: {best_config[0]}x\n") # type: ignore
        f.write(f"  插值方法: {best_config[1].capitalize()}\n") # type: ignore
        f.write(f"  PSNR值: {best_psnr:.2f} dB\n")
        
        f.write("\n六、分析结论:\n")
        f.write("-"*50 + "\n")
        f.write("  1. 随着下采样因子增大，信息丢失增多，PSNR值显著下降\n")
        f.write("  2. 双三次插值通常获得最好的重建质量，其次是双线性插值，最近邻插值效果最差\n")
        f.write("  3. 对于2倍下采样，使用双三次插值可获得较高质量的重建（PSNR > 30dB）\n")
        f.write("  4. 对于4倍及以上下采样，即使使用最好的插值方法，重建质量也会明显下降\n")
        f.write("  5. 由于人眼对亮度信息更敏感，保持Y通道不变可以保留主要视觉信息\n")
        f.write("  6. 色度通道下采样利用了人眼对色度细节不敏感的特性\n")
        f.write("  7. 实际应用中，通常采用4:2:0或4:2:2的色度采样格式，在保证视觉质量的同时压缩数据\n")
        
        f.write("\n七、文件输出清单:\n")
        f.write("-"*50 + "\n")
        f.write("  原始图像:\n")
        f.write("    - original_rgb.jpg\n")
        f.write("    - original_y_channel.jpg\n")
        f.write("    - original_cb_channel.jpg\n")
        f.write("    - original_cr_channel.jpg\n")
        f.write("  下采样通道:\n")
        for factor in downsample_factors:
            f.write(f"    - cb_downsampled_{factor}x.jpg\n")
            f.write(f"    - cr_downsampled_{factor}x.jpg\n")
        f.write("  重建图像:\n")
        for factor in downsample_factors:
            for method in interpolation_methods:
                f.write(f"    - reconstructed_{factor}x_{method}.jpg\n")
        f.write("  对比图像:\n")
        f.write("    - all_results_comparison.png\n")
        f.write("    - psnr_comparison.png\n")
        f.write("  结果文件:\n")
        f.write("    - psnr_results.txt\n")
        f.write("    - analysis_report.txt\n")
    
    print(f"分析报告已保存: {os.path.join(output_dir, 'analysis_report.txt')}")
    
    # 9. 打印最终总结
    print("\n" + "="*80)
    print("实验完成！所有结果已保存到:")
    print(f"  {output_dir}")
    print("="*80)
    
    print("\n生成的文件列表:")
    all_files = os.listdir(output_dir)
    for file in sorted(all_files):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.2f} KB)")

if __name__ == "__main__":
    main()