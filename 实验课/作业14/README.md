# Sinusoidal Position Encoding 与 RoPE 的实现与比较

## 项目简介

本项目实现了 Transformer 模型中的两种经典位置编码方法：
1. **Sinusoidal Position Encoding**（正余弦位置编码）
2. **RoPE (Rotary Position Embedding)**（旋转位置嵌入）

通过数值实验和可视化对比，深入分析两种方法的原理、差异及 RoPE 的优越性。

## 实验环境

- **操作系统**：Linux
- **Python版本**：3.x
- **依赖库**：
  ```
  numpy
  matplotlib
  scipy
  ```

## 项目结构

```
.
├──src
│   ├── main.py          # 主程序代码
├── README.md                   # 项目说明文档
├── output/                     # 实验结果输出目录
│   ├── rope_relative_property.png
│   └── position_encoding_comparison.png
└── 实验报告.docx               # 详细实验报告
```

## 核心功能实现

### 1. Sinusoidal Position Encoding

实现经典的 Transformer 位置编码，公式如下：

```python
def sinusoidal_position_encoding(seq_len, d_model):
    PE[pos, 2i] = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

### 2. 二维向量旋转

实现基本的二维旋转矩阵：

```python
def rotate_2d(vec, theta):
    rot_matrix = [[cosθ, -sinθ], [sinθ, cosθ]]
    return rot_matrix @ vec
```

### 3. 高维 RoPE

将相邻两维组成二维子空间分别旋转：

```python
def rope_apply(x, position):
    # 对每个二维子空间应用旋转
    for i in range(0, d_model, 2):
        angle = position * base^(-2i/d_model)
        # 旋转变换
```

### 4. E+pos vs RoPE 对比

对比两种输入注入方式对词向量的影响：
- **E+pos**：加法注入，改变向量模长
- **RoPE**：旋转注入，保持向量模长不变

### 5. 相对位置性质验证

通过数值实验验证 RoPE 的核心性质：**内积结果仅依赖于相对位置差**

## 实验结果摘要

### 1. Sinusoidal PE 输出示例

```
Sinusoidal PE (5 tokens, dim=6):
[[ 0.     1.     0.     1.     0.     1.   ]
 [ 0.841  0.54   0.046  0.999  0.002  1.   ]
 [ 0.909 -0.416  0.093  0.996  0.004  1.   ]
 [ 0.141 -0.99   0.139  0.99   0.006  1.   ]
 [-0.757 -0.654  0.185  0.983  0.009  1.   ]]
```

### 2. E+pos vs RoPE 对比

| 编码方式 | 向量模长 | 是否改变模长 | 语义独立性 |
|----------|----------|--------------|------------|
| 原始 embedding | 2.489 | — | 纯语义 |
| E+pos | 3.746 | ✓ 改变 | 语义+位置混合 |
| RoPE | 2.489 | ✗ 不变 | 语义独立 |

### 3. 相对位置性质验证

| 相对位置 | 内积均值 | 标准差 |
|----------|----------|--------|
| +2 | -2.9868 | 0.0000 |
| -3 | -2.7100 | 0.0000 |

**关键发现**：相同相对位置差的内积在不同绝对位置组合下完全相同（标准差为 0），严格证明了 RoPE 的相对位置性质。

## 核心结论

### RoPE 比 Sinusoidal PE 更巧妙的原因

1. **几何直观**：用旋转代替加法，语义作为"长度"，位置作为"角度"，自然解耦

2. **数学优美**：旋转矩阵的正交性保证模长不变，且满足 R_m^T R_n = R_{n-m}

3. **归纳偏置**：显式编码相对位置，符合注意力机制的"相对关系更重要"的直觉

4. **实践验证**：LLaMA、GPT-NeoX、PaLM 等主流大模型均采用 RoPE

## 可视化结果说明

### rope_relative_property.png
- **左图**：热力图显示不同绝对位置组合的内积值
- **右图**：误差棒图显示内积值仅依赖于相对位置差

### position_encoding_comparison.png
- **第一行**：Sinusoidal PE 和 RoPE 的编码热力图对比
- **第二行**：两种编码的相似性矩阵对比
