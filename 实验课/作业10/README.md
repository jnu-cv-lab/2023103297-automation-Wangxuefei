
# 基于 OpenCV 的局部特征检测、描述与图像匹配

## 项目简介

本实验基于OpenCV实现ORB特征检测、描述与图像匹配，通过RANSAC剔除错误匹配并利用单应矩阵完成目标定位。实验包含特征检测、特征匹配、RANSAC优化、目标定位及参数对比分析等完整流程。

## 文件结构

```
作业10/
├── box.png                    # 模板图像（目标物体）
├── box_in_scene.png           # 场景图像（包含目标）
├── 源
    ├── main.py                    # 主程序代码
└── output/                    # 输出结果目录
    ├── box_keypoints.png                 # 模板图特征点
    ├── box_in_scene_keypoints.png        # 场景图特征点
    ├── orb_all_matches.png               # 全部匹配结果
    ├── orb_best_50_matches.png           # 最佳50个匹配
    ├── ransac_inliers.png                # RANSAC内点匹配
    ├── ransac_outliers.png               # RANSAC外点匹配
    ├── box_localization_result.png       # 目标定位结果
    ├── nfeatures_comparison_analysis.png # 参数对比分析图
    ├── matching_results.txt              # 匹配结果报告
    ├── localization_report.txt           # 定位结果报告
    ├── nfeatures_experiment_report.txt   # 参数实验报告
    ├── ...
```

## 实验任务

| 任务 | 内容 | 主要输出 |
|------|------|---------|
| 任务1 | ORB特征点检测 | 特征点可视化图、关键点数量 |
| 任务2 | 特征匹配 | 匹配结果图、匹配距离分布 |
| 任务3 | RANSAC剔除误匹配 | 内点/外点匹配图、单应矩阵 |
| 任务4 | 目标定位 | 定位结果图、定位报告 |
| 任务6 | 参数对比实验 | 对比分析图、实验报告 |

## 关键结果

### 特征检测结果
- box.png 关键点：865个
- box_in_scene.png 关键点：1000个
- 描述子维度：32（256位二进制）

### 匹配与定位结果
- 总匹配数量：287对
- RANSAC内点：52个（18.12%）
- 定位结果：部分成功

### 参数对比结论

| nfeatures | 模板图关键点 | 场景图关键点 | 匹配数 | 内点数 | 内点比例 |
|-----------|------------|------------|--------|--------|----------|
| 500 | 453 | 500 | 148 | 32 | 21.62% |
| 1000 | 865 | 1000 | 287 | 52 | 18.12% |
| 2000 | 1589 | 1999 | 511 | 63 | 12.33% |

**结论**：增加nfeatures会提高匹配数量，但内点比例反而下降，说明特征点质量比数量更重要。

## 核心代码示例

```python
# ORB特征检测
orb = cv2.ORB_create(nfeatures=1000)
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)

# 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
匹配项 = sorted(匹配项, key=lambda x: x.距离)

# RANSAC剔除误匹配
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 目标定位
corners = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
transformed = cv2.perspectiveTransform(corners, H)
cv2.polylines(场景, [np.int32(转换后)], True, (0,255,0), 3)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| nfeatures | 1000 | 最多检测的特征点数量 |
| NORM_HAMMING | - | ORB二进制描述子的距离度量 |
| crossCheck | True | 启用交叉检验提高匹配质量 |
| RANSAC阈值 | 5.0 | 重投影误差阈值（像素） |

## 常见问题

**Q1：为什么内点比例偏低？**
- 场景中盒子存在较大角度旋转，透视变形严重
- 包装盒上重复纹理较多，导致误匹配增加

**Q2：如何提高定位精度？**
- 适当调整RANSAC阈值（3.0~5.0）
- 尝试nfeatures=500~800的中间值
- 使用KNN匹配+Lowe比率测试

**Q3：特征点越多越好吗？**
- 不一定。增加特征点会提高匹配数量，但也可能引入更多外点
- 实验显示nfeatures=2000时内点比例反而下降

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `*_keypoints.png` | 绿色圆圈标记特征点位置和方向 |
| `*_matches.png` | 彩色连线表示匹配关系 |
| `ransac_inliers.png` | 绿色连线表示通过几何验证的匹配 |
| `ransac_outliers.png` | 红色连线表示被剔除的错误匹配 |
| `*_localization_result.png` | 绿色四边形框标记检测到的目标位置 |
| `*.txt` | 详细的数值结果和统计报告 |

## 详细分析位于实验报告中
