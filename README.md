
# 图像处理基础项目

这是一个使用 OpenCV、NumPy 和 Matplotlib 进行图像处理的基础项目，演示了图像的读取、转换、裁剪和显示等基本操作。

## 📋 项目简介

本项目实现了以下图像处理功能：
- 读取图像文件
- 输出图像基本信息（尺寸、通道数、数据类型等）
- 将彩色图像转换为灰度图
- 使用 NumPy 切片裁剪图像左上角区域
- 保存处理后的图像
- 使用 Matplotlib 可视化显示处理结果

## 🛠️ 技术栈

- **Python 3.7+**
- **OpenCV** - 图像读取和处理
- **NumPy** - 数组操作和图像裁剪
- **Matplotlib** - 图像可视化显示

## 📁 项目结构

```
image-processing-project/
├── .vscode/                 # VS Code 配置文件
│   ├── settings.json        # 编辑器设置
│   └── launch.json          # 调试配置
├── images/
│   ├── input/               # 输入图像文件夹
│   │   └── test.jpg         # 测试图像
│   └── output/              # 输出图像文件夹
│       ├── .gitkeep         # 保持文件夹结构
│       ├── grayscale.jpg    # 灰度图输出
│       └── cropped_top_left.jpg  # 裁剪图像输出
├── src/
│   └── main.py              # 主程序文件
├── tests/                   # 测试文件夹（可选）
├── .gitignore               # Git 忽略文件
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.7 或更高版本
- pip 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/image-processing-project.git
cd image-processing-project
```

2. **创建虚拟环境（推荐）**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **准备测试图像**
- 将您的测试图像放入 `images/input/` 文件夹
- 确保文件名为 `test.jpg`（或修改代码中的文件名）

5. **运行程序**
```bash
python src/main.py
```

## 📝 代码说明

### 主要功能模块

#### 1. 图像读取
```python
img = cv2.imread(image_path)
```

#### 2. 获取图像信息
```python
height, width = img.shape[:2]
channels = img.shape[2] if len(img.shape) == 3 else 1
```

#### 3. 颜色空间转换
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

#### 4. NumPy 切片裁剪
```python
# 裁剪左上角 300x300 区域
cropped = img[0:crop_height, 0:crop_width, :]
```

#### 5. 图像保存
```python
cv2.imwrite(output_path, gray)
```

#### 6. 可视化显示
```python
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

## 📊 输出示例

运行程序后，控制台将输出：
```
读取图片: /path/to/images/input/test.jpg

==================================================
✅ 图片读取成功！
图像尺寸: 1920 x 1080
图像宽度: 1920 像素
图像高度: 1080 像素
通道数: 3
数据类型: uint8
==================================================

📐 裁剪信息:
裁剪区域大小: 300 x 300 像素
裁剪区域位置: 左上角 (0, 0)
裁剪后图像形状: (300, 300, 3)

✅ 裁剪图像已保存: /path/to/images/output/cropped_top_left.jpg

灰度图已保存: /path/to/images/output/grayscale.jpg
```

同时会显示一个包含三张图像的 Matplotlib 窗口：
- 原始图像
- 灰度图像
- 裁剪后的图像

## 🔧 自定义配置

### 修改裁剪区域大小
在 `main.py` 中修改 `crop_size` 变量：
```python
crop_size = 300  # 改为您需要的大小
```

### 修改输入图像路径
```python
image_path = os.path.join(current_dir, '..', 'images', 'input', 'your_image.jpg')
```

### 修改输出图像格式
```python
# 改为 PNG 格式
output_path = os.path.join(current_dir, '..', 'images', 'output', 'grayscale.png')
```

## 📦 依赖说明

创建 `requirements.txt` 文件：

```txt
opencv-python==4.8.1.78
numpy==1.24.3
matplotlib==3.7.2
```

## 🐛 常见问题

### 1. 图像文件不存在错误
**错误**: `错误: 文件不存在 - .../test.jpg`

**解决方案**: 
- 确保 `images/input/` 文件夹中存在 `test.jpg` 文件
- 检查文件路径是否正确
- 确认文件名大小写是否正确

### 2. 无法读取图像
**错误**: `错误: 无法读取图片`

**解决方案**:
- 确认图像格式是否被 OpenCV 支持（jpg、png、bmp 等）
- 检查图像文件是否损坏
- 确保图像路径不包含中文字符

### 3. Matplotlib 后端错误
**解决方案**: 代码中已设置 `matplotlib.use('TkAgg')`，如果仍有问题：
```python
# 尝试其他后端
matplotlib.use('Qt5Agg')
# 或
matplotlib.use('Agg')
```

### 4. 虚拟环境激活失败
**Windows PowerShell**:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📖 学习资源

- [OpenCV 官方文档](https://docs.opencv.org/)
- [NumPy 官方教程](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib 教程](https://matplotlib.org/stable/tutorials/index.html)

**⭐ 如果这个项目对您有帮助，请给个 Star！**
``
