import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建 output 文件夹
if not os.path.exists("output"):
    os.makedirs("output")

# ============================================================
# 1. 设计测试图
# ============================================================
def create_test_image():
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    # 矩形
    cv2.rectangle(img, (150, 100), (450, 300), (0, 0, 255), 3)
    # 圆
    cv2.circle(img, (300, 200), 60, (255, 0, 0), 3)
    # 平行线（水平）
    for y in [400, 450, 500]:
        cv2.line(img, (80, y), (520, y), (0, 255, 0), 3)
    # 垂直线
    for x in [120, 250, 380]:
        cv2.line(img, (x, 380), (x, 560), (255, 255, 0), 3)
    
    cv2.putText(img, "Rectangle", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(img, "Circle", (270, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(img, "Parallel Lines", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(img, "Vertical Lines", (50, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    return img


# ============================================================
# 2. 构造变换矩阵
# ============================================================
def get_similarity_matrix(angle_deg, scale, tx, ty):
    theta = np.radians(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.float32([
        [scale * cos, -scale * sin, tx],
        [scale * sin, scale * cos, ty]
    ])


def get_affine_matrix(a, b, c, d, tx, ty):
    return np.float32([[a, b, tx], [c, d, ty]])


def get_perspective_matrix(pts1, pts2):
    A = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1] / Vt[-1, -1]
    return h.reshape(3, 3).astype(np.float32)


# ============================================================
# 3. 应用变换
# ============================================================
def apply_transform(img, M, is_perspective=False):
    h, w = img.shape[:2]
    
    if is_perspective:
        corners = np.float32([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
        transformed = M @ corners
        transformed = transformed / transformed[2]
        x_min, x_max = int(np.min(transformed[0])), int(np.max(transformed[0]))
        y_min, y_max = int(np.min(transformed[1])), int(np.max(transformed[1]))
        
        translation = np.float32([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        M_adjusted = translation @ M
        new_w, new_h = x_max - x_min, y_max - y_min
        
        return cv2.warpPerspective(img, M_adjusted, (new_w, new_h))
    else:
        corners = np.float32([[0,0], [w,0], [w,h], [0,h]])
        transformed = cv2.transform(corners.reshape(1,4,2), M).reshape(-1,2)
        x_min, x_max = int(np.min(transformed[:,0])), int(np.max(transformed[:,0]))
        y_min, y_max = int(np.min(transformed[:,1])), int(np.max(transformed[:,1]))
        
        M_adjusted = M.copy()
        M_adjusted[0,2] -= x_min
        M_adjusted[1,2] -= y_min
        new_w, new_h = x_max - x_min, y_max - y_min
        
        return cv2.warpAffine(img, M_adjusted, (new_w, new_h))


# ============================================================
# 4. 精确的自动纸张检测
# ============================================================
def auto_detect_paper_accurate(img):
    """
    精确检测纸张角点
    使用：灰度化 -> 高斯模糊 -> Canny边缘检测 -> 轮廓查找 -> 四边形拟合
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. 自适应阈值二值化
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 2)
    
    # 3. 形态学操作：闭运算连接断开的边缘
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. 膨胀，填充内部空洞
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 5. 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 6. 按面积排序，取最大的几个
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    
    best_quad = None
    best_area = 0
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 多边形逼近
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 如果是四边形且面积合理
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)
    
    if best_quad is not None:
        return sort_corners(best_quad)
    
    # 如果没找到四边形，使用最小外接矩形
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    corners = cv2.boxPoints(rect)
    return sort_corners(np.int32(corners))


def sort_corners(corners):
    """将四个角点排序为：左上、右上、右下、左下"""
    corners = np.float32(corners)
    center = np.mean(corners, axis=0)
    
    def get_angle(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])
    
    sorted_corners = sorted(corners, key=get_angle, reverse=True)
    tl, tr, br, bl = sorted_corners
    
    # 确保顺序正确
    if tl[0] > tr[0]:
        tl, tr = tr, tl
    if bl[0] > br[0]:
        bl, br = br, bl
    if tl[1] > bl[1]:
        tl, bl = bl, tl
    if tr[1] > br[1]:
        tr, br = br, tr
    
    return np.float32([tl, tr, br, bl])


# ============================================================
# 5. 透视校正
# ============================================================
def correct_perspective_auto(image_path):
    """自动检测并校正透视畸变"""
    print("\n" + "=" * 60)
    print("透视校正 (cv2.warpPerspective)")
    print("=" * 60)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：找不到 {image_path}")
        return None
    
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")
    
    # 自动检测角点
    src_pts = auto_detect_paper_accurate(img)
    
    if src_pts is None:
        print("自动检测失败！")
        return None
    
    print("\n检测到的角点:")
    names = ["左上", "右上", "右下", "左下"]
    for i, pt in enumerate(src_pts):
        print(f"  {names[i]}: ({pt[0]:.0f}, {pt[1]:.0f})")
    
    # 标记角点
    img_marked = img.copy()
    for i, pt in enumerate(src_pts):
        cv2.circle(img_marked, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        cv2.putText(img_marked, names[i], (int(pt[0])+15, int(pt[1])-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.polylines(img_marked, [np.int32(src_pts)], True, (0, 255, 0), 3)
    cv2.imwrite("output/corners_detected.png", img_marked)
    print("\n保存: output/corners_detected.png")
    
    # 计算目标尺寸（A4比例）
    width = int(max(np.linalg.norm(src_pts[0] - src_pts[1]),
                    np.linalg.norm(src_pts[2] - src_pts[3])))
    height = int(width * 297 / 210)
    
    dst_pts = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
    
    # 透视校正
    M = get_perspective_matrix(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, M, (width, height))
    
    cv2.imwrite("output/corrected.png", corrected)
    print("保存: output/corrected.png")
    
    # 显示对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (with distortion)", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected Corners", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[2].set_title("After Perspective Correction", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("output/correction_comparison.png", dpi=150)
    plt.show()
    print("保存: output/correction_comparison.png")
    
    print("\n✓ 透视校正完成！")
    return corrected


# ============================================================
# 6. 主程序
# ============================================================
def main():
    print("=" * 60)
    print("实验：几何变换性质研究")
    print("=" * 60)
    
    # 创建测试图
    print("\n[1] 创建测试图...")
    test_img = create_test_image()
    cv2.imwrite("output/0_original.png", test_img)
    print("    保存: output/0_original.png")
    
    # 相似变换
    print("\n[2] 相似变换")
    M_sim = get_similarity_matrix(25, 0.8, 40, 30)
    sim_img = apply_transform(test_img, M_sim, is_perspective=False)
    cv2.imwrite("output/1_similarity.png", sim_img)
    print("    保存: output/1_similarity.png")
    
    # 仿射变换
    print("\n[3] 仿射变换")
    M_aff = get_affine_matrix(1.2, 0.5, 0.3, 1.1, 50, 40)
    aff_img = apply_transform(test_img, M_aff, is_perspective=False)
    cv2.imwrite("output/2_affine.png", aff_img)
    print("    保存: output/2_affine.png")
    
    # 透视变换
    print("\n[4] 透视变换")
    h, w = test_img.shape[:2]
    src_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])
    dst_pts = np.float32([[120,40], [w-120,30], [w-70,h-30], [70,h-30]])
    M_per = get_perspective_matrix(src_pts, dst_pts)
    per_img = apply_transform(test_img, M_per, is_perspective=True)
    cv2.imwrite("output/3_perspective.png", per_img)
    print("    保存: output/3_perspective.png")
    
    # 对比图
    print("\n[5] 生成对比图...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Original", fontsize=12)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("Similarity (warpAffine)", fontsize=12)
    axes[0,1].axis('off')
    
    axes[1,0].imshow(cv2.cvtColor(aff_img, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title("Affine (warpAffine)", fontsize=12)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(cv2.cvtColor(per_img, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title("Perspective (warpPerspective)", fontsize=12)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig("output/4_comparison.png", dpi=150)
    plt.show()
    print("    保存: output/4_comparison.png")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    main()
    correct_perspective_auto("a4_photo.jpg")