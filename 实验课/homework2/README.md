# YCbCr 色彩空间下采样与插值重建实验

## 项目简介

本项目实现了基于 YCbCr 色彩空间的图像下采样与插值重建实验。通过对比不同下采样因子和插值方法对图像重建质量的影响，验证了人眼对色度信息不敏感的特性，为图像压缩编码提供理论依据。

### 实验原理

- **YCbCr 色彩空间**：将 RGB 图像转换到 YCbCr 空间，其中 Y 表示亮度分量，Cb 和 Cr 表示色度分量
- **色度下采样**：人眼对亮度信息敏感，但对色度细节不敏感，因此可以对色度通道进行下采样
- **插值重建**：使用不同插值方法将下采样的色度通道恢复至原始尺寸
- **质量评估**：通过 PSNR（峰值信噪比）客观评估重建图像质量

## 功能特性

-  RGB 到 YCbCr 色彩空间转换
-  支持 2x、4x、8x 下采样因子
-  三种插值方法：最近邻、双线性、双三次
-  PSNR 客观质量评估
-  自动保存所有处理结果
-  生成对比图和 PSNR 分析图表
-  生成详细的实验分析报告

## 环境要求

### 系统要求
- Python 3.7+
- OpenCV 4.x
- NumPy
- Matplotlib

### 安装依赖

```bash
pip install opencv-python numpy matplotlib
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 项目结构

```
.
├── src
│   ├── main.py              # 主程序文件
├── images/
│   ├── input/             # 输入图像目录
│   │   └── test.jpg       # 测试图像
│   └── output/            # 输出结果目录
├── .vscode/               # VS Code 配置文件
│   ├── settings.json      # 项目设置
│   └── launch.json        # 调试配置
├── README.md              # 项目说明文档
└── requirements.txt       # Python 依赖列表
```

## 使用方法

### 1. 准备测试图像

将待处理的图像放置在 `images/input/` 目录下，命名为 `test.jpg`

### 2. 运行程序

```bash
python main.py
```

### 3. 查看结果

程序运行后，所有结果将保存在 `images/output/` 目录下：

#### 原始图像
- `original_rgb.jpg` - 原始 RGB 图像
- `original_y_channel.jpg` - Y 通道（亮度）
- `original_cb_channel.jpg` - Cb 通道（蓝色色度）
- `original_cr_channel.jpg` - Cr 通道（红色色度）

#### 下采样通道
- `cb_downsampled_{factor}x.jpg` - 下采样后的 Cb 通道
- `cr_downsampled_{factor}x.jpg` - 下采样后的 Cr 通道

#### 重建图像
- `reconstructed_{factor}x_{method}.jpg` - 重建的 RGB 图像
  - factor: 2, 4, 8
  - method: nearest, bilinear, bicubic

#### 分析结果
- `all_results_comparison.png` - 所有结果对比图
- `psnr_comparison.png` - PSNR 对比曲线图
- `psnr_results.txt` - PSNR 结果表格
- `analysis_report.txt` - 详细实验分析报告

## 实验结果示例

### PSNR 结果对比表

| Factor | Nearest | Bilinear | Bicubic |
|--------|---------|----------|---------|
| 2x     | 47.10 dB | 50.32 dB | 50.18 dB |
| 4x     | 41.25 dB | 43.65 dB | 43.39 dB |
| 8x     | 37.55 dB | 39.59 dB | 39.23 dB |

### 分析结论

1. **下采样因子影响**：随着下采样因子增大，信息丢失增多，PSNR 值显著下降
2. **插值方法对比**：
   - 双三次插值（Bicubic）效果最佳
   - 双线性插值（Bilinear）次之
   - 最近邻插值（Nearest）效果最差
3. **实用建议**：
   - 2x 下采样 + 双三次插值可获得高质量重建（PSNR > 30dB）
   - 4x 下采样可接受，但质量明显下降
   - 8x 下采样质量较差，不推荐使用

## 代码说明

### 核心函数

#### `calculate_psnr(img1, img2)`
计算两幅图像的峰值信噪比

#### `downsample_channel(channel, factor)`
对单个通道进行下采样

#### `upsample_channel(channel, original_shape, method)`
对单个通道进行上采样恢复

#### `reconstruct_image(y_channel, cb_channel, cr_channel)`
使用 Y、Cb、Cr 通道重建 RGB 图像

## 自定义配置

### 修改图像路径

在 `main.py` 中修改以下变量：

```python
# 输入图像路径
image_path = '/your/path/to/image.jpg'

# 输出目录
output_dir = '/your/path/to/output'
```

### 修改下采样因子

```python
# 修改此列表以调整下采样因子
downsample_factors = [2, 4, 8]  # 可添加其他值如 16
```

### 修改插值方法

```python
# 修改此列表以调整插值方法
interpolation_methods = ['nearest', 'bilinear', 'bicubic']
```

## VS Code 调试配置

项目包含完整的 VS Code 调试配置：

- **Python: 运行主程序** - 直接运行程序
- **Python: 调试主程序** - 调试模式运行
- **Python: 运行当前文件** - 调试当前打开的文件
- **Python: 带参数运行主程序** - 带命令行参数运行
- **Python: 单元测试** - 运行测试用例

按 `F5` 开始调试，或在代码中设置断点（点击行号左侧）

## 注意事项

1. **图像格式**：确保输入图像为常见的图片格式（jpg、png 等）
2. **路径权限**：确保输出目录有写入权限
3. **内存占用**：处理大尺寸图像时注意内存使用
4. **色彩空间**：OpenCV 使用 BGR 格式，代码已做相应转换

## 扩展应用

- **视频压缩**：将色度下采样技术应用于视频编码（如 JPEG、MPEG）
- **实时图像处理**：优化算法实现实时处理
- **深度学习**：使用神经网络进行超分辨率重建
- **多尺度分析**：结合小波变换进行多尺度处理

## 配套的 requirements.txt 文件

为了方便其他用户安装依赖，建议同时创建 `requirements.txt`：

```txt
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
```
