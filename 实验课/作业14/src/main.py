import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import os

# 创建输出目录
output_dir = "/home/wxf81/作业12/作业14/output"
os.makedirs(output_dir, exist_ok=True)

# ================== 1. Sinusoidal Position Encoding ==================
def sinusoidal_position_encoding(seq_len, d_model):
    """生成经典的Sinusoidal位置编码 (Vaswani et al.)"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度
    return pe

# ================== 2. 二维向量旋转 ==================
def rotate_2d(vec, theta):
    """将二维向量逆时针旋转theta弧度"""
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
    return np.dot(rot_matrix, vec)

# 演示二维旋转
vec2d = np.array([1.0, 0.0])
theta = np.pi / 2  # 90度
rotated = rotate_2d(vec2d, theta)
print(f"2D Rotation: {vec2d} rotate {theta:.2f} rad -> {rotated}")  # 应接近 [0,1]

# ================== 3. 高维RoPE实现 ==================
def rope_embedding(seq_len, d_model, base=10000.0):
    """
    生成RoPE所需的旋转角度矩阵 (每个token对应一组旋转角度)
    返回: (seq_len, d_model/2, 2, 2) 的块对角旋转矩阵集合
    实际使用时，通常将查询/键的相邻两维构成一个复数: (x_{2i} + i*x_{2i+1}) * e^{i*theta}
    这里直接生成旋转变换矩阵以展示原理。
    """
    assert d_model % 2 == 0, "d_model must be even"
    # 每个二维子空间的旋转频率
    freq = base ** (-2 * (np.arange(d_model // 2) + 1) / d_model)  # 论文公式
    positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len,1)
    angles = positions * freq  # (seq_len, d_model/2)
    
    # 构建每个token的块对角旋转矩阵 (seq_len, d_model, d_model)
    rope_matrices = np.zeros((seq_len, d_model, d_model))
    for t in range(seq_len):
        blocks = []
        for dim in range(d_model // 2):
            cos_t = np.cos(angles[t, dim])
            sin_t = np.sin(angles[t, dim])
            # 2x2 旋转矩阵
            rot = np.array([[cos_t, -sin_t],
                            [sin_t,  cos_t]])
            blocks.append(rot)
        rope_matrices[t] = block_diag(*blocks)
    return rope_matrices

# 或者更高效的实现：直接对向量对进行旋转 (不构建大矩阵)
def rope_apply(x, position, base=10000.0):
    """
    对单个向量x (shape: d_model) 应用位置position的RoPE旋转。
    实际使用中使用复数乘法，这里直接演示旋转效果。
    """
    d_model = len(x)
    assert d_model % 2 == 0
    d_half = d_model // 2
    freq = base ** (-2 * (np.arange(d_half) + 1) / d_model)
    angles = position * freq
    x_rotated = np.zeros_like(x)
    for i in range(0, d_model, 2):
        cos_t = np.cos(angles[i//2])
        sin_t = np.sin(angles[i//2])
        x_rotated[i] = x[i] * cos_t - x[i+1] * sin_t
        x_rotated[i+1] = x[i] * sin_t + x[i+1] * cos_t
    return x_rotated

# ================== 4. 对比 E+pos (加法) 与 RoPE 的输入方式 ==================
def compare_input_encodings():
    """
    E+pos: embedding加上位置编码 (sinusoidal)
    RoPE: embedding按位置旋转 (不改变词向量模长，但改变方向)
    """
    d_model = 8
    seq_len = 4
    # 模拟词嵌入 (随机)
    np.random.seed(42)
    embeddings = np.random.randn(seq_len, d_model)
    
    # Sinusoidal 位置编码 (加法)
    sin_pe = sinusoidal_position_encoding(seq_len, d_model)
    e_plus_pos = embeddings + sin_pe
    
    # RoPE 旋转 (不改变向量范数)
    rope_embeddings = np.zeros_like(embeddings)
    for t in range(seq_len):
        rope_embeddings[t] = rope_apply(embeddings[t], t)
    
    print("\n=== Input Encoding Comparison ===")
    print("Original embedding[0]:", embeddings[0])
    print("E+pos[0]:", e_plus_pos[0])
    print("RoPE[0]:", rope_embeddings[0])
    print("Norm: original=%.3f, E+pos=%.3f, RoPE=%.3f" % (np.linalg.norm(embeddings[0]),
                                                          np.linalg.norm(e_plus_pos[0]),
                                                          np.linalg.norm(rope_embeddings[0])))
    # RoPE保持模长不变
    return e_plus_pos, rope_embeddings

# ================== 5. 数值实验：验证RoPE相对位置性质 ==================
def experiment_relative_position():
    """
    关键性质：RoPE后，两个token的内积只依赖于它们的相对位置差。
    验证: < RoPE(x, m), RoPE(y, n) > = f(x,y, m-n)
    """
    d_model = 8
    np.random.seed(123)
    x = np.random.randn(d_model)  # 词向量1
    y = np.random.randn(d_model)  # 词向量2
    
    max_pos = 20
    inner_prods = np.zeros((max_pos, max_pos))
    
    for pos1 in range(max_pos):
        for pos2 in range(max_pos):
            x_rope = rope_apply(x, pos1)
            y_rope = rope_apply(y, pos2)
            inner_prods[pos1, pos2] = np.dot(x_rope, y_rope)
    
    # 看不同相对位置差的对角线是否一致
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(inner_prods, cmap='viridis')
    axes[0].set_title("Inner product after RoPE")
    axes[0].set_xlabel("pos2")
    axes[0].set_ylabel("pos1")
    plt.colorbar(im, ax=axes[0])
    
    # 提取相对位置差的效果: 内积仅依赖于 (pos1 - pos2)
    diff_values = {}
    for pos1 in range(max_pos):
        for pos2 in range(max_pos):
            d = pos1 - pos2
            diff_values.setdefault(d, []).append(inner_prods[pos1, pos2])
    
    relative_diffs = sorted(diff_values.keys())
    means = [np.mean(diff_values[d]) for d in relative_diffs]
    stds = [np.std(diff_values[d]) for d in relative_diffs]
    
    axes[1].errorbar(relative_diffs, means, yerr=stds, fmt='o-', capsize=3)
    axes[1].set_title("Inner product vs. Relative position (m-n)")
    axes[1].set_xlabel("Relative position (pos1 - pos2)")
    axes[1].set_ylabel("Inner product")
    axes[1].grid(True)
    
    plt.tight_layout()
    # 保存图片到指定目录
    save_path = os.path.join(output_dir, "rope_relative_property.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n图片已保存到: {save_path}")
    plt.show()
    
    print("\n=== Relative Position Property ===")
    print("For relative position +2: mean=%.4f, std=%.4f" % (np.mean(diff_values[2]), np.std(diff_values[2])))
    print("For relative position -3: mean=%.4f, std=%.4f" % (np.mean(diff_values[-3]), np.std(diff_values[-3])))
    print("不同绝对位置但相同相对差的内积几乎相同 (std很小)，验证了RoPE的相对位置性质。")

# ================== 额外：可视化Sinusoidal PE和RoPE ==================
def visualize_position_encodings():
    """可视化对比两种位置编码方式"""
    seq_len = 50
    d_model = 128
    
    # Sinusoidal PE
    sin_pe = sinusoidal_position_encoding(seq_len, d_model)
    
    # 计算RoPE后的位置向量（使用单位向量来纯粹展示位置编码效果）
    rope_pos_vectors = np.zeros((seq_len, d_model))
    for t in range(seq_len):
        # 使用单位向量来单独展示位置编码效果
        unit_vec = np.ones(d_model) / np.sqrt(d_model)
        rope_pos_vectors[t] = rope_apply(unit_vec, t)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sinusoidal PE 热力图
    im1 = axes[0, 0].imshow(sin_pe[:30, :30], aspect='auto', cmap='RdBu')
    axes[0, 0].set_title('Sinusoidal PE (first 30 tokens, first 30 dims)')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Position')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RoPE 旋转向量热力图
    im2 = axes[0, 1].imshow(rope_pos_vectors[:30, :30], aspect='auto', cmap='RdBu')
    axes[0, 1].set_title('RoPE applied to unit vector (first 30 tokens, first 30 dims)')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Position')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 位置编码的相似性矩阵对比
    sin_sim = np.dot(sin_pe, sin_pe.T)
    rope_sim = np.dot(rope_pos_vectors, rope_pos_vectors.T)
    
    im3 = axes[1, 0].imshow(sin_sim[:30, :30], cmap='hot')
    axes[1, 0].set_title('Sinusoidal PE similarity matrix')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Position')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(rope_sim[:30, :30], cmap='hot')
    axes[1, 1].set_title('RoPE similarity matrix (unit vector)')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Position')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "position_encoding_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n对比图已保存到: {save_path}")
    plt.show()

if __name__ == "__main__":
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 1. Sinusoidal PE example
    pe_example = sinusoidal_position_encoding(5, 6)
    print("Sinusoidal PE (5 tokens, dim=6):\n", pe_example.round(3))
    
    # 2. 二维旋转 demo already done
    
    # 3. 高维RoPE demo
    rope_matrices_4x8 = rope_embedding(seq_len=2, d_model=4)  # 2 tokens, d_model=4
    print("\nRoPE matrix for token 0 (4x4 block diag):\n", rope_matrices_4x8[0].round(3))
    
    # 4. Compare E+pos vs RoPE
    compare_input_encodings()
    
    # 5. Numerical experiment: relative position property
    experiment_relative_position()
    
    # 6. Additional visualization
    visualize_position_encodings()
    
    print(f"\n所有图片已保存到: {output_dir}")