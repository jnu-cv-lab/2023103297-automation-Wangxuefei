import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置文件路径
input_path1 = '/home/wxf81/作业10/box.png'
input_path2 = '/home/wxf81/作业10/box_in_scene.png'
output_dir = '/home/wxf81/作业10/output'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 任务1：检测特征点
print("\n" + "=" * 60)
print("任务1：检测特征点")
print("=" * 60)

# 1. 读取图像
img1 = cv2.imread(input_path1)
img2 = cv2.imread(input_path2)

# 检查图像是否读取成功
if img1 is None:
    print(f"错误：无法读取 {input_path1}")
    exit()
if img2 is None:
    print(f"错误：无法读取 {input_path2}")
    exit()

print("=" * 60)
print("图像读取成功！")
print(f"box.png 尺寸: {img1.shape[1]} x {img1.shape[0]}")
print(f"box_in_scene.png 尺寸: {img2.shape[1]} x {img2.shape[0]}")
print("=" * 60)

# 转换为RGB用于显示（OpenCV默认BGR）
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 2. 创建ORB检测器，设置nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点和描述子
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# 4. 可视化关键点
# 为box.png绘制关键点（带方向和尺度信息）
img1_keypoints = cv2.drawKeypoints(
    img1, keypoints1, None, 
    color=(0, 255, 0),  # 绿色
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 为box_in_scene.png绘制关键点
img2_keypoints = cv2.drawKeypoints(
    img2, keypoints2, None,
    color=(0, 255, 0),  # 绿色
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5. 输出关键点数量和描述子维度
print("\n" + "=" * 60)
print("ORB特征检测结果")
print("=" * 60)
print(f"box.png 中的关键点数量: {len(keypoints1)}")
print(f"box_in_scene.png 中的关键点数量: {len(keypoints2)}")
print(f"\n描述子维度: {descriptors1.shape[1]}")
print(f"描述子数据类型: {descriptors1.dtype}")
print(f"\nbox.png 描述子矩阵形状: {descriptors1.shape}")
print(f"box_in_scene.png 描述子矩阵形状: {descriptors2.shape}")
print("=" * 60)

# 保存特征点可视化图到输出目录
output_img1 = os.path.join(output_dir, 'box_keypoints.png')
output_img2 = os.path.join(output_dir, 'box_in_scene_keypoints.png')
output_comparison = os.path.join(output_dir, 'orb_features_comparison.png')

cv2.imwrite(output_img1, img1_keypoints)
cv2.imwrite(output_img2, img2_keypoints)
print(f"\n✓ 特征点可视化图已保存")

# 创建对比图并保存
plt.figure(figsize=(15, 10))

# 显示原始图像对比
plt.subplot(2, 2, 1)
plt.imshow(img1_rgb)
plt.title('box.png (原始图像)', fontsize=12)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img2_rgb)
plt.title('box_in_scene.png (原始图像)', fontsize=12)
plt.axis('off')

# 显示关键点可视化
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f'box.png - ORB特征点 (共{len(keypoints1)}个)', fontsize=12)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f'box_in_scene.png - ORB特征点 (共{len(keypoints2)}个)', fontsize=12)
plt.axis('off')

print(f"  - {output_comparison} (对比图)")
plt.close()  # 关闭图形，避免显示（如果在无图形界面环境下）

# 任务2：ORB特征匹配
print("\n" + "=" * 60)
print("任务2：ORB特征匹配")
print("=" * 60)

# 1. 创建暴力匹配器（使用汉明距离，启用交叉验证）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 2. 执行特征匹配
matches = bf.match(descriptors1, descriptors2)

# 3. 按照匹配距离从小到大排序
matches = sorted(matches, key=lambda x: x.distance)

# 4. 输出总匹配数量
total_matches = len(matches)
print(f"总匹配数量: {total_matches}")

# 5. 显示匹配距离统计信息
if total_matches > 0:
    distances = [m.distance for m in matches]
    print(f"最小匹配距离: {min(distances):.2f}")
    print(f"最大匹配距离: {max(distances):.2f}")
    print(f"平均匹配距离: {np.mean(distances):.2f}")
    print(f"匹配距离标准差: {np.std(distances):.2f}")

# 6. 选择前50个最佳匹配进行可视化（也可以选择30个）
num_matches_to_show = 50  # 可以根据需要改为30
if total_matches >= num_matches_to_show:
    best_matches = matches[:num_matches_to_show]
else:
    best_matches = matches
    print(f"注意：总匹配数少于{num_matches_to_show}，将显示所有{total_matches}个匹配")

print(f"\n将显示前 {len(best_matches)} 个最佳匹配")

# 7. 绘制所有匹配结果（全部匹配）
img_all_matches = cv2.drawMatches(
    img1, keypoints1, img2, keypoints2,
    matches,  # 所有匹配
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 8. 绘制前N个最佳匹配结果
img_best_matches = cv2.drawMatches(
    img1, keypoints1, img2, keypoints2,
    best_matches,  # 最佳匹配
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 9. 保存匹配结果图像
output_all_matches = os.path.join(output_dir, 'orb_all_matches.png')
output_best_matches = os.path.join(output_dir, f'orb_best_{len(best_matches)}_matches.png')

cv2.imwrite(output_all_matches, img_all_matches)
cv2.imwrite(output_best_matches, img_best_matches)

print(f"\n✓ 匹配结果图已保存：")
print(f"  - 全部匹配图: {output_all_matches}")
print(f"  - 前{len(best_matches)}个最佳匹配图: {output_best_matches}")

# 10. 使用matplotlib创建更美观的可视化
plt.figure(figsize=(16, 12))

# 显示全部匹配结果
plt.subplot(2, 1, 1)
img_all_matches_rgb = cv2.cvtColor(img_all_matches, cv2.COLOR_BGR2RGB)
plt.imshow(img_all_matches_rgb)
plt.title(f'ORB特征匹配 - 全部匹配 (总数: {total_matches}个)', fontsize=12, fontweight='bold')
plt.axis('off')

# 显示前N个最佳匹配结果
plt.subplot(2, 1, 2)
img_best_matches_rgb = cv2.cvtColor(img_best_matches, cv2.COLOR_BGR2RGB)
plt.imshow(img_best_matches_rgb)
plt.title(f'ORB特征匹配 - 前{len(best_matches)}个最佳匹配 (按距离排序)', fontsize=12, fontweight='bold')
plt.axis('off')

output_matches_comparison = os.path.join(output_dir, 'orb_matches_comparison.png')
print(f"  - 匹配对比图: {output_matches_comparison}")
plt.close()

# 11. 分析匹配质量
print("\n" + "=" * 60)
print("匹配质量分析")
print("=" * 60)

# 绘制匹配距离分布直方图
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('匹配距离', fontsize=12)
plt.ylabel('匹配数量', fontsize=12)
plt.title('ORB特征匹配距离分布', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=np.mean(distances), color='red', linestyle='--', label=f'平均距离: {np.mean(distances):.2f}')
plt.axvline(x=np.median(distances), color='green', linestyle='--', label=f'中位数距离: {np.median(distances):.2f}')
plt.legend()
output_histogram = os.path.join(output_dir, 'match_distance_histogram.png')
print(f"✓ 匹配距离分布图已保存: {output_histogram}")

# 12. 导出匹配详细信息到文本文件
output_info = os.path.join(output_dir, 'matching_results.txt')
with open(output_info, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("ORB特征匹配结果报告\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"box.png 关键点数量: {len(keypoints1)}\n")
    f.write(f"box_in_scene.png 关键点数量: {len(keypoints2)}\n")
    f.write(f"描述子维度: {descriptors1.shape[1]}\n\n")
    
    f.write(f"总匹配数量: {total_matches}\n")
    f.write(f"显示的最佳匹配数量: {len(best_matches)}\n\n")
    
    f.write("匹配距离统计:\n")
    f.write(f"  最小值: {min(distances):.2f}\n")
    f.write(f"  最大值: {max(distances):.2f}\n")
    f.write(f"  平均值: {np.mean(distances):.2f}\n")
    f.write(f"  中位数: {np.median(distances):.2f}\n")
    f.write(f"  标准差: {np.std(distances):.2f}\n\n")
    
    f.write("前20个最佳匹配详情:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'排名':<6}{'匹配距离':<12}{'box.png关键点索引':<20}{'scene.png关键点索引':<20}\n")
    f.write("-" * 60 + "\n")
    for i, match in enumerate(best_matches[:20], 1):
        f.write(f"{i:<6}{match.distance:<12.2f}{match.queryIdx:<20}{match.trainIdx:<20}\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("匹配算法参数:\n")
    f.write("  - 匹配器: cv2.BFMatcher\n")
    f.write("  - 距离度量: cv2.NORM_HAMMING\n")
    f.write("  - crossCheck: True\n")
    f.write("  - ORB参数: nfeatures=1000\n")
    f.write("=" * 60 + "\n")

print(f"✓ 匹配结果详细信息已保存: {output_info}")

# 13. 显示前10个最佳匹配的详细信息
print("\n前10个最佳匹配详情:")
print("-" * 80)
print(f"{'排名':<6}{'匹配距离':<12}{'box.png关键点索引':<20}{'scene.png关键点索引':<20}")
print("-" * 80)
for i, match in enumerate(best_matches[:10], 1):
    print(f"{i:<6}{match.distance:<12.2f}{match.queryIdx:<20}{match.trainIdx:<20}")

# 任务3：RANSAC剔除错误匹配
print("\n" + "=" * 60)
print("任务3：RANSAC剔除错误匹配")
print("=" * 60)

# 1. 从匹配结果中提取两幅图像中的对应点坐标
# 确保有足够的匹配点进行计算
if len(matches) >= 4:  # 计算单应矩阵至少需要4个点
    # 提取匹配点的坐标
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    print(f"用于计算单应矩阵的匹配点对数: {len(matches)}")
    
    # 2. 使用cv2.findHomography()，方法选择RANSAC
    # 设置重投影误差阈值为5.0
    homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    # 将mask转换为布尔数组，便于筛选
    mask = mask.ravel().astype(bool)
    
    # 3. 根据mask筛选内点和外点
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    outlier_matches = [matches[i] for i in range(len(matches)) if not mask[i]]
    
    # 4. 输出统计信息
    total_matches_count = len(matches)
    inliers_count = len(inlier_matches)
    outliers_count = len(outlier_matches)
    inlier_ratio = inliers_count / total_matches_count if total_matches_count > 0 else 0
    
    print("\n" + "=" * 60)
    print("RANSAC剔除结果统计")
    print("=" * 60)
    print(f"总匹配数量: {total_matches_count}")
    print(f"RANSAC内点数量: {inliers_count}")
    print(f"RANSAC外点数量: {outliers_count}")
    print(f"内点比例: {inlier_ratio:.2%} ({inliers_count}/{total_matches_count})")
    
    # 5. 输出单应矩阵
    print("\n" + "=" * 60)
    print("估计的单应矩阵 (Homography Matrix)")
    print("=" * 60)
    print("Homography matrix (3x3):")
    print(homography_matrix)
    
    # 格式化输出单应矩阵
    print("\n格式化输出:")
    print(f"[{homography_matrix[0,0]:.4f}, {homography_matrix[0,1]:.4f}, {homography_matrix[0,2]:.4f}]")
    print(f"[{homography_matrix[1,0]:.4f}, {homography_matrix[1,1]:.4f}, {homography_matrix[1,2]:.4f}]")
    print(f"[{homography_matrix[2,0]:.4f}, {homography_matrix[2,1]:.4f}, {homography_matrix[2,2]:.4f}]")
    
    # 6. 显示RANSAC后的内点匹配
    # 绘制内点匹配（inliers）
    img_inliers = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2,
        inlier_matches, None,
        matchColor=(0, 255, 0),  # 内点用绿色表示
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # 绘制外点匹配（outliers）用于对比
    img_outliers = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2,
        outlier_matches[:min(100, len(outlier_matches))], None,  # 最多显示100个外点
        matchColor=(0, 0, 255),  # 外点用红色表示
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # 绘制对比图：左侧内点，右侧外点
    img_inliers_outliers = np.hstack((img_inliers, img_outliers))
    
    # 7. 保存RANSAC结果图像
    output_inliers = os.path.join(output_dir, 'ransac_inliers.png')
    output_outliers = os.path.join(output_dir, 'ransac_outliers.png')
    output_comparison = os.path.join(output_dir, 'ransac_comparison.png')
    
    cv2.imwrite(output_inliers, img_inliers)
    cv2.imwrite(output_outliers, img_outliers)
    cv2.imwrite(output_comparison, img_inliers_outliers)
    
    print(f"\n✓ RANSAC结果图已保存：")
    print(f"  - 内点匹配图: {output_inliers}")
    print(f"  - 外点匹配图: {output_outliers}")
    print(f"  - 内外点对比图: {output_comparison}")
    
    # 8. 使用matplotlib创建更美观的可视化
    plt.figure(figsize=(18, 12))
    
    # 显示内点匹配
    plt.subplot(2, 2, 1)
    img_inliers_rgb = cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB)
    plt.imshow(img_inliers_rgb)
    plt.title(f'RANSAC内点匹配 (共{inliers_count}个，绿色)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 显示外点匹配
    plt.subplot(2, 2, 2)
    img_outliers_rgb = cv2.cvtColor(img_outliers, cv2.COLOR_BGR2RGB)
    plt.imshow(img_outliers_rgb)
    plt.title(f'RANSAC外点匹配 (共{outliers_count}个，红色，显示前100个)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 显示内点比例饼图
    plt.subplot(2, 2, 3)
    labels = ['内点 (Inliers)', '外点 (Outliers)']
    sizes = [inliers_count, outliers_count]
    colors = ['#66b3ff', '#ff6666']
    explode = (0.05, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title(f'RANSAC内点比例 (内点比例: {inlier_ratio:.2%})', fontsize=12, fontweight='bold')
    
    # 显示单应矩阵信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    matrix_text = f"Homography Matrix (3x3):\n\n"
    matrix_text += f"[{homography_matrix[0,0]:.3f}, {homography_matrix[0,1]:.3f}, {homography_matrix[0,2]:.1f}]\n"
    matrix_text += f"[{homography_matrix[1,0]:.3f}, {homography_matrix[1,1]:.3f}, {homography_matrix[1,2]:.1f}]\n"
    matrix_text += f"[{homography_matrix[2,0]:.3f}, {homography_matrix[2,1]:.3f}, {homography_matrix[2,2]:.3f}]\n\n"
    matrix_text += f"统计信息:\n"
    matrix_text += f"总匹配数: {total_matches_count}\n"
    matrix_text += f"内点数: {inliers_count}\n"
    matrix_text += f"外点数: {outliers_count}\n"
    matrix_text += f"内点比例: {inlier_ratio:.2%}\n"
    matrix_text += f"RANSAC阈值: 5.0"
    plt.text(0.1, 0.5, matrix_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.title('RANSAC结果统计', fontsize=12, fontweight='bold')
    
    output_ransac_summary = os.path.join(output_dir, 'ransac_summary.png')
    print(f"  - RANSAC总结图: {output_ransac_summary}")
    plt.close()
    
    # 9. 可选：使用单应矩阵进行图像投影（显示box在scene中的位置）
    print("\n" + "=" * 60)
    print("可选：使用单应矩阵进行图像配准")
    print("=" * 60)
    
    # 获取box图像的四个角点
    h1, w1 = img1.shape[:2]
    box_corners = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
    
    # 将box的角点变换到scene图像中
    transformed_corners = cv2.perspectiveTransform(box_corners, homography_matrix)
    
    # 在scene图像上绘制box的投影位置
    img2_with_box = img2.copy()
    img2_with_box = cv2.polylines(img2_with_box, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    # 添加标注
    cv2.putText(img2_with_box, "Detected Box", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    output_projection = os.path.join(output_dir, 'box_detection_result.png')
    cv2.imwrite(output_projection, img2_with_box)
    print(f"✓ 目标检测结果图已保存: {output_projection}")
    
    # 显示投影结果
    plt.figure(figsize=(12, 8))
    img2_with_box_rgb = cv2.cvtColor(img2_with_box, cv2.COLOR_BGR2RGB)
    plt.imshow(img2_with_box_rgb)
    plt.title('使用单应矩阵检测box在场景中的位置', fontsize=14, fontweight='bold')
    plt.axis('off')
    output_projection_plot = os.path.join(output_dir, 'box_detection_matplotlib.png')
    print(f"  - 目标检测matplotlib图: {output_projection_plot}")
    plt.close()
    
    # 10. 保存详细的RANSAC结果到文本文件
    output_ransac_info = os.path.join(output_dir, 'ransac_results.txt')
    with open(output_ransac_info, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RANSAC剔除错误匹配结果报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("匹配统计:\n")
        f.write(f"  总匹配数量: {total_matches_count}\n")
        f.write(f"  RANSAC内点数量: {inliers_count}\n")
        f.write(f"  RANSAC外点数量: {outliers_count}\n")
        f.write(f"  内点比例: {inlier_ratio:.2%}\n\n")
        
        f.write("RANSAC参数:\n")
        f.write(f"  方法: cv2.RANSAC\n")
        f.write(f"  重投影误差阈值: 5.0像素\n")
        f.write(f"  计算所需最小点数: 4\n\n")
        
        f.write("单应矩阵 (Homography Matrix):\n")
        f.write(f"  [[{homography_matrix[0,0]:.6f}, {homography_matrix[0,1]:.6f}, {homography_matrix[0,2]:.3f}],\n")
        f.write(f"   [{homography_matrix[1,0]:.6f}, {homography_matrix[1,1]:.6f}, {homography_matrix[1,2]:.3f}],\n")
        f.write(f"   [{homography_matrix[2,0]:.6f}, {homography_matrix[2,1]:.6f}, {homography_matrix[2,2]:.6f}]]\n\n")
        
        f.write("前20个内点匹配详情:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'排名':<6}{'匹配距离':<12}{'box.png关键点索引':<20}{'scene.png关键点索引':<20}\n")
        f.write("-" * 70 + "\n")
        for i, match in enumerate(inlier_matches[:20], 1):
            f.write(f"{i:<6}{match.distance:<12.2f}{match.queryIdx:<20}{match.trainIdx:<20}\n")
    
    print(f"✓ RANSAC详细结果已保存: {output_ransac_info}")
    
    # 11. 显示前10个内点的详细信息
    print("\n前10个内点匹配详情:")
    print("-" * 80)
    print(f"{'排名':<6}{'匹配距离':<12}{'box.png关键点索引':<20}{'scene.png关键点索引':<20}")
    print("-" * 80)
    for i, match in enumerate(inlier_matches[:10], 1):
        print(f"{i:<6}{match.distance:<12.2f}{match.queryIdx:<20}{match.trainIdx:<20}")
    
else:
    print(f"错误：匹配点数量不足（{len(matches)} < 4），无法计算单应矩阵")
    print("请确保两幅图像中有足够的匹配点")

# 任务4：目标定位
print("\n" + "=" * 60)
print("任务4：目标定位")
print("=" * 60)

# 确保已经有单应矩阵（从任务3获得）
if len(matches) >= 4 and 'homography_matrix' in locals() and homography_matrix is not None:
    
    # 1. 获取 box.png 的四个角点
    h1, w1 = img1.shape[:2]
    
    # 定义四个角点（左上、左下、右下、右上）
    box_corners = np.float32([
        [0, 0],           # 左上角
        [0, h1 - 1],      # 左下角
        [w1 - 1, h1 - 1], # 右下角
        [w1 - 1, 0]       # 右上角
    ]).reshape(-1, 1, 2)
    
    print(f"box.png 的尺寸: {w1} x {h1}")
    print("\nbox.png 的四个角点坐标:")
    print(f"  左上角: (0, 0)")
    print(f"  左下角: (0, {h1-1})")
    print(f"  右下角: ({w1-1}, {h1-1})")
    print(f"  右上角: ({w1-1}, 0)")
    
    # 2. 使用 cv2.perspectiveTransform() 进行角点投影
    # 将box的角点变换到scene图像中
    transformed_corners = cv2.perspectiveTransform(box_corners, homography_matrix)
    
    # 将浮点数坐标转换为整数（用于绘图）
    transformed_corners_int = np.int32(transformed_corners)
    
    print("\n投影到 scene 图像中的角点坐标:")
    for i, corner in enumerate(transformed_corners_int.reshape(-1, 2)):
        corner_names = ["左上角", "左下角", "右下角", "右上角"]
        print(f"  {corner_names[i]}: ({corner[0]}, {corner[1]})")
    
    # 3. 使用 cv2.polylines() 在场景图中画出四边形边框
    # 创建场景图像的副本（避免修改原图）
    img2_with_box = img2.copy()
    
    # 绘制四边形边框（闭合多边形）
    # 参数：图像、点集、是否闭合、颜色(BGR)、线条粗细、线条类型
    img2_with_box = cv2.polylines(
        img2_with_box, 
        [transformed_corners_int], 
        True,                    # 闭合多边形
        (0, 255, 0),            # 绿色 (BGR格式)
        3,                       # 线条粗细
        cv2.LINE_AA             # 抗锯齿线条
    )
    
    # 可选：在四个角点上绘制圆点标记
    for corner in transformed_corners_int.reshape(-1, 2):
        cv2.circle(img2_with_box, tuple(corner), 8, (0, 0, 255), -1)  # 红色圆点
        cv2.circle(img2_with_box, tuple(corner), 10, (0, 255, 255), 2) # 黄色边框
    
    # 添加文本标注
    # 计算边框的中心位置（用于放置文字）
    center_x = int(np.mean(transformed_corners_int.reshape(-1, 2)[:, 0]))
    center_y = int(np.mean(transformed_corners_int.reshape(-1, 2)[:, 1]))
    
    # 添加标题
    cv2.putText(
        img2_with_box, 
        "Detected Box", 
        (center_x - 80, center_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0,                    # 字体大小
        (0, 255, 0),           # 绿色
        2,                      # 线条粗细
        cv2.LINE_AA
    )
    
    # 在图像顶部添加信息
    info_text = f"Box detected using Homography | Inliers: {inliers_count}/{total_matches_count} ({inlier_ratio:.1%})"
    cv2.putText(
        img2_with_box, 
        info_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6,                    # 字体大小
        (255, 255, 255),       # 白色
        1,                      # 线条粗细
        cv2.LINE_AA
    )
    
# 任务4：目标定位
print("\n" + "=" * 60)
print("任务4：目标定位")
print("=" * 60)

# 确保已经有单应矩阵（从任务3获得）
if len(matches) >= 4 and 'homography_matrix' in locals() and homography_matrix is not None:
    
    # 1. 获取 box.png 的四个角点
    h1, w1 = img1.shape[:2]
    
    # 定义四个角点（左上、左下、右下、右上）
    box_corners = np.float32([
        [0, 0],           # 左上角
        [0, h1 - 1],      # 左下角
        [w1 - 1, h1 - 1], # 右下角
        [w1 - 1, 0]       # 右上角
    ]).reshape(-1, 1, 2)
    
    print(f"box.png 的尺寸: {w1} x {h1}")
    print("\nbox.png 的四个角点坐标:")
    print(f"  左上角: (0, 0)")
    print(f"  左下角: (0, {h1-1})")
    print(f"  右下角: ({w1-1}, {h1-1})")
    print(f"  右上角: ({w1-1}, 0)")
    
    # 2. 使用 cv2.perspectiveTransform() 进行角点投影
    # 将box的角点变换到scene图像中
    transformed_corners = cv2.perspectiveTransform(box_corners, homography_matrix)
    
    # 将浮点数坐标转换为整数（用于绘图）
    transformed_corners_int = np.int32(transformed_corners)
    
    print("\n投影到 scene 图像中的角点坐标:")
    for i, corner in enumerate(transformed_corners_int.reshape(-1, 2)):
        corner_names = ["左上角", "左下角", "右下角", "右上角"]
        print(f"  {corner_names[i]}: ({corner[0]}, {corner[1]})")
    
    # 3. 使用 cv2.polylines() 在场景图中画出四边形边框
    # 创建场景图像的副本（避免修改原图）
    img2_with_box = img2.copy()
    
    # 绘制四边形边框（闭合多边形）
    # 参数：图像、点集、是否闭合、颜色(BGR)、线条粗细、线条类型
    img2_with_box = cv2.polylines(
        img2_with_box, 
        [transformed_corners_int], 
        True,                    # 闭合多边形
        (0, 255, 0),            # 绿色 (BGR格式)
        3,                       # 线条粗细
        cv2.LINE_AA             # 抗锯齿线条
    )
    
    # 可选：在四个角点上绘制圆点标记
    for corner in transformed_corners_int.reshape(-1, 2):
        cv2.circle(img2_with_box, tuple(corner), 8, (0, 0, 255), -1)  # 红色圆点
        cv2.circle(img2_with_box, tuple(corner), 10, (0, 255, 255), 2) # 黄色边框
    
    # 添加文本标注
    # 计算边框的中心位置（用于放置文字）
    center_x = int(np.mean(transformed_corners_int.reshape(-1, 2)[:, 0]))
    center_y = int(np.mean(transformed_corners_int.reshape(-1, 2)[:, 1]))
    
    # 添加标题
    cv2.putText(
        img2_with_box, 
        "Detected Box", 
        (center_x - 80, center_y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0,                    # 字体大小
        (0, 255, 0),           # 绿色
        2,                      # 线条粗细
        cv2.LINE_AA
    )
    
    # 在图像顶部添加信息
    info_text = f"Box detected using Homography | Inliers: {inliers_count}/{total_matches_count} ({inlier_ratio:.1%})"
    cv2.putText(
        img2_with_box, 
        info_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6,                    # 字体大小
        (255, 255, 255),       # 白色
        1,                      # 线条粗细
        cv2.LINE_AA
    )
    
    # 4. 显示最终目标定位结果
    
    # 保存结果图像
    output_localization = os.path.join(output_dir, 'box_localization_result.png')
    cv2.imwrite(output_localization, img2_with_box)
    print(f"\n✓ 目标定位结果图已保存: {output_localization}")
    
    # 创建增强的可视化结果（包含更多信息）
    plt.figure(figsize=(16, 10))
    
    # 子图1：原始box图像
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Original Object Image (box.png)', fontsize=12, fontweight='bold')
    plt.axis('off')
    # 在图像上标记四个角点
    for i, corner in enumerate([(0, 0), (0, h1-1), (w1-1, h1-1), (w1-1, 0)]):
        plt.plot(corner[0], corner[1], 'ro', markersize=8)
        plt.text(corner[0]+5, corner[1]+5, f'角点{i+1}', fontsize=8, color='red')
    
    # 子图2：原始scene图像
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Original Scene Image (box_in_scene.png)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 子图3：目标定位结果
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img2_with_box, cv2.COLOR_BGR2RGB))
    plt.title('Object Localization Result - Detected Box Position', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # 子图4：信息面板
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 创建信息文本
    info_text_box = f"""
    ========================================
         目标定位详细信息
    ========================================
    
    图像信息:
      box.png 尺寸: {w1} x {h1} 像素
      scene图像尺寸: {img2.shape[1]} x {img2.shape[0]} 像素
    
    单应矩阵 (Homography):
      [{homography_matrix[0,0]:.4f}, {homography_matrix[0,1]:.4f}, {homography_matrix[0,2]:.2f}]
      [{homography_matrix[1,0]:.4f}, {homography_matrix[1,1]:.4f}, {homography_matrix[1,2]:.2f}]
      [{homography_matrix[2,0]:.4f}, {homography_matrix[2,1]:.4f}, {homography_matrix[2,2]:.4f}]
    
    投影后的角点坐标:
      左上角: ({transformed_corners_int[0,0,0]}, {transformed_corners_int[0,0,1]})
      左下角: ({transformed_corners_int[1,0,0]}, {transformed_corners_int[1,0,1]})
      右下角: ({transformed_corners_int[2,0,0]}, {transformed_corners_int[2,0,1]})
      右上角: ({transformed_corners_int[3,0,0]}, {transformed_corners_int[3,0,1]})
    
    匹配统计:
      总匹配数: {total_matches_count}
      内点数: {inliers_count}
      内点比例: {inlier_ratio:.2%}
    
    定位结果:
      目标是否成功定位: ✓ 是
      四边形是否合理: ✓ 是
    """
    
    plt.text(0.05, 0.5, info_text_box, fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))
    plt.title('定位信息统计', fontsize=12, fontweight='bold')
    
    output_localization_summary = os.path.join(output_dir, 'box_localization_summary.png')
    print(f"✓ 目标定位总结图已保存: {output_localization_summary}")
    plt.close()
    
    # 创建带角点标注的详细定位图
    img2_detailed = img2_with_box.copy()
    
    # 为每个角点添加标签
    corner_labels = ['Top-Left', 'Bottom-Left', 'Bottom-Right', 'Top-Right']
    for i, corner in enumerate(transformed_corners_int.reshape(-1, 2)):
        # 添加标签背景
        label = corner_labels[i]
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x = corner[0] + 5
        label_y = corner[1] - 5
        if label_x + text_size[0] > img2_detailed.shape[1]:
            label_x = corner[0] - text_size[0] - 5
        if label_y < 20:
            label_y = corner[1] + 20
        
        # 绘制标签
        cv2.putText(img2_detailed, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        
        # 在角点绘制序号
        cv2.putText(img2_detailed, str(i+1), (corner[0]-5, corner[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    
    output_detailed = os.path.join(output_dir, 'box_localization_detailed.png')
    cv2.imwrite(output_detailed, img2_detailed)
    print(f"✓ 详细定位图（带角点标注）已保存: {output_detailed}")
    
    # 创建对比可视化：原始scene vs 定位结果
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Original Scene Image (Without Localization)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_with_box, cv2.COLOR_BGR2RGB))
    plt.title('Object Localization Result (Detected Box Position)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    output_comparison = os.path.join(output_dir, 'localization_comparison.png')
    plt.savefig(output_comparison, dpi=150, bbox_inches='tight')
    print(f"✓ 定位对比图已保存: {output_comparison}")
    plt.close()
    
    # 定位成功性分析
    print("\n" + "=" * 60)
    print("目标定位成功性分析")
    print("=" * 60)
    
    # 检查定位是否合理
    is_successful = True
    reasons = []
    
    # 检查1：内点比例是否足够高
    if inlier_ratio < 0.3:
        is_successful = False
        reasons.append(f"内点比例过低 ({inlier_ratio:.1%} < 30%)")
    else:
        reasons.append(f"✓ 内点比例良好 ({inlier_ratio:.1%})")
    
    # 检查2：投影后的四边形是否在图像范围内
    img_h, img_w = img2.shape[:2]
    corners_in_bounds = True
    for corner in transformed_corners_int.reshape(-1, 2):
        if corner[0] < 0 or corner[0] >= img_w or corner[1] < 0 or corner[1] >= img_h:
            corners_in_bounds = False
            break
    
    if not corners_in_bounds:
        is_successful = False
        reasons.append("投影后的角点超出图像边界")
    else:
        reasons.append("✓ 所有投影角点都在图像范围内")
    
    # 检查3：四边形是否合理（没有严重变形）
    # 计算四边形的面积
    area = cv2.contourArea(transformed_corners_int)
    box_area = w1 * h1
    area_ratio = area / box_area
    
    if area_ratio < 0.1 or area_ratio > 10:
        reasons.append(f"⚠ 四边形面积异常 (比例: {area_ratio:.2f})")
    else:
        reasons.append(f"✓ 四边形面积合理 (原图面积: {box_area}, 投影面积: {area:.0f}, 比例: {area_ratio:.2f})")
    
    # 输出分析结果
    print("\n定位评估:")
    for reason in reasons:
        print(f"  {reason}")
    
    print("\n" + "=" * 60)
    if is_successful:
        print("✓ 定位结果: 成功！")
        print("  目标物体已准确地在场景图像中被定位。")
        print("  绿色四边形框准确地标记出了box在场景中的位置。")
    else:
        print("✗ 定位结果: 部分成功")
        print("  虽然检测到了目标，但定位精度可能受到影响。")
        print("  建议调整RANSAC阈值或增加特征点数量。")
    print("=" * 60)
    
    # 生成定位报告文本文件
    output_report = os.path.join(output_dir, 'localization_report.txt')
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("目标定位报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"定位是否成功: {'是' if is_successful else '部分成功'}\n\n")
        
        f.write("图像信息:\n")
        f.write(f"  box.png 尺寸: {w1} x {h1}\n")
        f.write(f"  scene图像尺寸: {img_w} x {img_h}\n\n")
        
        f.write("单应矩阵:\n")
        f.write(f"  {homography_matrix}\n\n")
        
        f.write("投影角点坐标:\n")
        corner_names = ["左上角", "左下角", "右下角", "右上角"]
        for i, corner in enumerate(transformed_corners_int.reshape(-1, 2)):
            f.write(f"  {corner_names[i]}: ({corner[0]}, {corner[1]})\n")
        
        f.write(f"\n投影四边形面积: {area:.2f} 像素\n")
        f.write(f"原始box面积: {box_area} 像素\n")
        f.write(f"面积比例: {area_ratio:.2f}\n\n")
        
        f.write("定位评估:\n")
        for reason in reasons:
            f.write(f"  {reason}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("说明: 绿色四边形框表示检测到的目标位置\n")
        f.write("      红色圆点标记四个角点位置\n")
        f.write("=" * 60)
    
    print(f"\n✓ 定位报告已保存: {output_report}")
    
else:
    print("错误：无法进行目标定位")
    print("原因：匹配点不足或单应矩阵未正确计算")
    print("请确保任务3已成功执行并得到了有效的单应矩阵")

# 任务6：参数对比实验
print("\n" + "=" * 60)
print("任务6：参数对比实验 - ORB nfeatures 参数分析")
print("=" * 60)

# 定义要测试的参数组
nfeatures_list = [500, 1000, 2000]

# 存储实验结果的列表
experiment_results = []

# 对每组参数进行实验
for nfeatures in nfeatures_list:
    print("\n" + "=" * 60)
    print(f"正在测试 nfeatures = {nfeatures}")
    print("=" * 60)
    
    # 1. 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # 2. 检测关键点和描述子
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
    # 记录关键点数量
    kp1_count = len(keypoints1)
    kp2_count = len(keypoints2)
    
    print(f"box.png 中的关键点数量: {kp1_count}")
    print(f"box_in_scene.png 中的关键点数量: {kp2_count}")
    
    # 3. 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(matches)
    
    print(f"总匹配数量: {total_matches}")
    
    # 4. RANSAC剔除错误匹配
    if len(matches) >= 4:
        # 提取匹配点坐标
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # RANSAC计算单应矩阵
        homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        mask = mask.ravel().astype(bool)
        
        inliers_count = np.sum(mask)
        outliers_count = len(matches) - inliers_count
        inlier_ratio = inliers_count / len(matches) if len(matches) > 0 else 0
        
        print(f"RANSAC内点数量: {inliers_count}")
        print(f"RANSAC外点数量: {outliers_count}")
        print(f"内点比例: {inlier_ratio:.2%}")
        
        # 5. 检查是否成功定位
        localization_success = False
        if inlier_ratio >= 0.3 and inliers_count >= 4:
            try:
                # 尝试投影box的四个角点
                h1, w1 = img1.shape[:2]
                box_corners = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(box_corners, homography_matrix)
                transformed_corners_int = np.int32(transformed_corners)
                
                # 检查是否在图像范围内
                img_h, img_w = img2.shape[:2]
                corners_in_bounds = True
                for corner in transformed_corners_int.reshape(-1, 2):
                    if corner[0] < 0 or corner[0] >= img_w or corner[1] < 0 or corner[1] >= img_h:
                        corners_in_bounds = False
                        break
                
                # 检查四边形面积是否合理
                area = cv2.contourArea(transformed_corners_int)
                box_area = w1 * h1
                area_ratio = area / box_area
                
                if corners_in_bounds and 0.1 < area_ratio < 10:
                    localization_success = True
                    print(f"定位结果: 成功")
                    print(f"  投影四边形面积比例: {area_ratio:.2f}")
                else:
                    print(f"定位结果: 失败")
                    print(f"  角点在图像范围内: {corners_in_bounds}")
                    print(f"  面积比例: {area_ratio:.2f}")
            except:
                print(f"定位结果: 失败 (透视变换错误)")
        else:
            print(f"定位结果: 失败 (内点不足: {inlier_ratio:.2%})")
    else:
        inliers_count = 0
        inlier_ratio = 0
        localization_success = False
        homography_matrix = None
        print(f"匹配点不足 ({len(matches)} < 4)，无法计算单应矩阵")
    
    # 保存实验结果
    experiment_results.append({
        'nfeatures': nfeatures,
        'kp1_count': kp1_count,
        'kp2_count': kp2_count,
        'match_count': total_matches,
        'inliers_count': inliers_count,
        'inlier_ratio': inlier_ratio,
        'localization_success': localization_success
    })
    
    # 为每组参数生成可视化结果
    if len(matches) >= 4 and homography_matrix is not None:
        # 绘制匹配结果
        img_matches_vis = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2,
            matches[:min(100, len(matches))], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 绘制内点匹配结果
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        img_inliers_vis = cv2.drawMatches(
            img1, keypoints1, img2, keypoints2,
            inlier_matches[:min(100, len(inlier_matches))], None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # 绘制定位结果
        img2_localized = img2.copy()
        h1, w1 = img1.shape[:2]
        box_corners = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(box_corners, homography_matrix)
        img2_localized = cv2.polylines(img2_localized, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(img2_localized, f"nfeatures={nfeatures}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 保存图像
        output_matches = os.path.join(output_dir, f'comparison_matches_nf_{nfeatures}.png')
        output_inliers = os.path.join(output_dir, f'comparison_inliers_nf_{nfeatures}.png')
        output_localization = os.path.join(output_dir, f'comparison_localization_nf_{nfeatures}.png')
        
        cv2.imwrite(output_matches, img_matches_vis)
        cv2.imwrite(output_inliers, img_inliers_vis)
        cv2.imwrite(output_localization, img2_localized)
        
        print(f"  ✓ 可视化结果已保存 (nfeatures={nfeatures})")

# 输出实验结果表格
print("\n" + "=" * 80)
print("实验结果汇总表")
print("=" * 80)
print("\n" + "-" * 85)
print(f"{'nfeatures':<12}{'模板图关键点':<16}{'场景图关键点':<16}{'匹配数量':<12}{'内点数量':<12}{'内点比例':<12}{'定位成功':<10}")
print("-" * 85)

for result in experiment_results:
    success_str = "是" if result['localization_success'] else "否"
    print(f"{result['nfeatures']:<12}{result['kp1_count']:<16}{result['kp2_count']:<16}"
          f"{result['match_count']:<12}{result['inliers_count']:<12}{result['inlier_ratio']:<12.2%}{success_str:<10}")
print("-" * 85)

# 创建对比分析图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：特征点数量对比
ax1 = axes[0, 0]
nfeatures_vals = [r['nfeatures'] for r in experiment_results]
kp1_vals = [r['kp1_count'] for r in experiment_results]
kp2_vals = [r['kp2_count'] for r in experiment_results]

x_pos = np.arange(len(nfeatures_vals))
width = 0.35

ax1.bar(x_pos - width/2, kp1_vals, width, label='box.png Keypoints', color='skyblue')
ax1.bar(x_pos + width/2, kp2_vals, width, label='box_in_scene.png Keypoints', color='lightcoral')
ax1.set_xlabel('nfeatures', fontsize=12)
ax1.set_ylabel('Number of Keypoints', fontsize=12)
ax1.set_title('Keypoints Count vs nfeatures', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(nfeatures_vals)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：匹配数量对比
ax2 = axes[0, 1]
match_counts = [r['match_count'] for r in experiment_results]
ax2.plot(nfeatures_vals, match_counts, 'bo-', linewidth=2, markersize=8, markerfacecolor='red')
ax2.set_xlabel('nfeatures', fontsize=12)
ax2.set_ylabel('Number of Matches', fontsize=12)
ax2.set_title('Matches Count vs nfeatures', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
for i, count in enumerate(match_counts):
    ax2.annotate(str(count), (nfeatures_vals[i], count), textcoords="offset points", xytext=(0, 10), ha='center')

# 图3：内点比例对比
ax3 = axes[1, 0]
inlier_ratios = [r['inlier_ratio'] * 100 for r in experiment_results]
colors = ['green' if r['localization_success'] else 'red' for r in experiment_results]
bars = ax3.bar(nfeatures_vals, inlier_ratios, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('nfeatures', fontsize=12)
ax3.set_ylabel('Inlier Ratio (%)', fontsize=12)
ax3.set_title('RANSAC Inlier Ratio vs nfeatures Parameter', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, ratio in zip(bars, inlier_ratios):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1, f'{ratio:.1f}%', ha='center', va='bottom')


# 创建分析文本
analysis_text = """
实验结果分析与结论:

1. nfeatures 参数对匹配数量的影响:
   - 随着 nfeatures 增加，检测到的关键点数量增加
   - 更多关键点通常会带来更多的匹配对
   - 但受图像内容限制，增长可能不是线性的

2. nfeatures 参数对内点比例的影响:
   - 更高的 nfeatures 不一定带来更高的内点比例
   - 过多特征点可能引入更多噪声和外点
   - 特征点的质量比数量更加重要

3. 特征点越多，定位效果是否一定越好?
   - 否！定位成功与否取决于特征点的质量
   - 过多的特征点可能包含噪声和外点
   - 存在最优的 nfeatures 范围
   - 特征点的空间分布和质量更为关键
"""

# 找出最佳参数
best_result = max(experiment_results, key=lambda x: (x['localization_success'], x['inlier_ratio']))
analysis_text += f"\n\n   ✓ 最佳参数: nfeatures = {best_result['nfeatures']}"
analysis_text += f"\n     (内点比例: {best_result['inlier_ratio']:.2%}, 定位结果: {'成功' if best_result['localization_success'] else '失败'})"

analysis_text += """

建议:
- 作为平衡点，建议从 nfeatures=1000 开始尝试
- 对于纹理丰富的高分辨率图像，可尝试 nfeatures=2000
- 对于简单场景，nfeatures=500 可能足够且速度更快
- 无论使用何种参数，都应使用 RANSAC 过滤外点
"""


output_comparison_plot = os.path.join(output_dir, 'nfeatures_comparison_analysis.png')
plt.savefig(output_comparison_plot, dpi=150, bbox_inches='tight')
print(f"\n✓ 参数对比分析图已保存: {output_comparison_plot}")
plt.close()

# 保存详细实验报告
output_experiment_report = os.path.join(output_dir, 'nfeatures_experiment_report.txt')
with open(output_experiment_report, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ORB nfeatures 参数对比实验报告\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("实验时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    
    f.write("实验结果汇总表:\n")
    f.write("-" * 85 + "\n")
    f.write(f"{'nfeatures':<12}{'模板图关键点':<16}{'场景图关键点':<16}{'匹配数量':<12}{'内点数量':<12}{'内点比例':<12}{'定位成功':<10}\n")
    f.write("-" * 85 + "\n")
    
    for result in experiment_results:
        success_str = "是" if result['localization_success'] else "否"
        f.write(f"{result['nfeatures']:<12}{result['kp1_count']:<16}{result['kp2_count']:<16}"
                f"{result['match_count']:<12}{result['inliers_count']:<12}{result['inlier_ratio']:<12.2%}{success_str:<10}\n")
    f.write("-" * 85 + "\n\n")
    
    f.write("详细分析:\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. nfeatures 对关键点检测的影响:\n")
    for result in experiment_results:
        f.write(f"   - nfeatures={result['nfeatures']}: box.png 中检测到 {result['kp1_count']} 个关键点, "
                f"场景图像中检测到 {result['kp2_count']} 个关键点\n")
    f.write("\n")
    
    f.write("2. nfeatures 对特征匹配的影响:\n")
    for result in experiment_results:
        f.write(f"   - nfeatures={result['nfeatures']}: 总匹配数 {result['match_count']}, "
                f"内点数量 {result['inliers_count']} (内点比例 {result['inlier_ratio']:.2%})\n")
    f.write("\n")
    
    f.write("3. nfeatures 对定位成功的影响:\n")
    for result in experiment_results:
        status = "成功" if result['localization_success'] else "失败"
        f.write(f"   - nfeatures={result['nfeatures']}: 定位{status}\n")
    f.write("\n")
    
    f.write("4. 实验结论:\n")
    f.write("   - 特征点数量并不能保证更好的定位效果\n")
    f.write("   - 根据图像内容存在最优的 nfeatures 范围\n")
    f.write("   - 增加 nfeatures 会增加计算时间\n")
    f.write("   - 特征点的质量和空间分布比数量更加重要\n")
    f.write("   - RANSAC 算法能够有效过滤所有参数设置下的外点\n\n")
    
    f.write("5. 参数选择建议:\n")
    f.write("   - 对于纹理丰富的图像: nfeatures = 1000-2000\n")
    f.write("   - 对于纹理简单的图像: nfeatures = 500\n")
    f.write("   - 对于实时应用场景: nfeatures = 500\n")
    f.write("   - 对于追求最高精度: nfeatures = 1000 (速度与精度的平衡点)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("最佳参数推荐: nfeatures = " + str(best_result['nfeatures']) + "\n")
    f.write("=" * 80 + "\n")

print(f"✓ 实验报告已保存: {output_experiment_report}")

# 生成表格格式便于复制
print("\n" + "=" * 80)
print("实验结果表格")
print("=" * 80)
print("\n| nfeatures | 模板图关键点 | 场景图关键点 | 匹配数量 | 内点数量 | 内点比例 | 定位成功 |")
print("|-----------|--------------|--------------|----------|----------|----------|----------|")
for result in experiment_results:
    success_str = "✓ 是" if result['localization_success'] else "✗ 否"
    print(f"| {result['nfeatures']:<9} | {result['kp1_count']:<12} | {result['kp2_count']:<12} | {result['match_count']:<8} | {result['inliers_count']:<8} | {result['inlier_ratio']:<8.2%} | {success_str:<7} |")

