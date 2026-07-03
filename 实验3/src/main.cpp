#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * 获取当前可执行文件所在目录
 * @return 当前目录路径
 */
std::string getCurrentDirectory()
{
    try
    {
        return fs::current_path().string();
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "获取当前目录失败: " << e.what() << std::endl;
        return "";
    }
}

/**
 * 打印图像信息
 * @param img 输入图像
 * @param name 图像名称
 */
void printImageInfo(const cv::Mat &img, const std::string &name)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "📷 " << name << " 图像信息:" << std::endl;
    std::cout << "图像尺寸: " << img.cols << " x " << img.rows << std::endl;
    std::cout << "宽度: " << img.cols << " 像素" << std::endl;
    std::cout << "高度: " << img.rows << " 像素" << std::endl;
    std::cout << "通道数: " << img.channels() << std::endl;
    std::cout << "数据类型: " << img.type() << std::endl;
    std::cout << "深度: " << img.depth() << std::endl;
    std::cout << "总像素数: " << img.total() << std::endl;
    std::cout << "================================================\n"
              << std::endl;
}

/**
 * 裁剪图像左上角区域
 * @param img 输入图像
 * @param cropSize 裁剪尺寸
 * @return 裁剪后的图像
 */
cv::Mat cropTopLeft(const cv::Mat &img, int cropSize)
{
    int cropHeight = std::min(cropSize, img.rows);
    int cropWidth = std::min(cropSize, img.cols);

    // 定义感兴趣区域（ROI）
    cv::Rect roi(0, 0, cropWidth, cropHeight);
    cv::Mat cropped = img(roi).clone(); // 克隆以创建独立副本

    std::cout << "\n 裁剪信息:" << std::endl;
    std::cout << "裁剪区域大小: " << cropWidth << " x " << cropHeight << " 像素" << std::endl;
    std::cout << "裁剪区域位置: 左上角 (0, 0)" << std::endl;
    std::cout << "裁剪后图像尺寸: " << cropped.cols << " x " << cropped.rows << std::endl;
    std::cout << "裁剪后通道数: " << cropped.channels() << std::endl;

    return cropped;
}

/**
 * 显示多图像窗口
 * @param original 原始图像
 * @param gray 灰度图像
 * @param cropped 裁剪图像
 */
void displayImages(const cv::Mat &original, const cv::Mat &gray, const cv::Mat &cropped)
{
    // 创建窗口并调整大小
    cv::namedWindow("图像处理结果", cv::WINDOW_NORMAL);
    cv::resizeWindow("图像处理结果", 1200, 400);

    // 创建画布，用于显示三张图像
    int totalWidth = original.cols + gray.cols + cropped.cols;
    int maxHeight = std::max({original.rows, gray.rows, cropped.rows});

    cv::Mat canvas = cv::Mat::zeros(maxHeight, totalWidth, original.type());

    // 放置原始图像
    original.copyTo(canvas(cv::Rect(0, 0, original.cols, original.rows)));

    // 放置灰度图像（需要转换为BGR才能正确显示）
    cv::Mat grayBGR;
    cv::cvtColor(gray, grayBGR, cv::COLOR_GRAY2BGR);
    grayBGR.copyTo(canvas(cv::Rect(original.cols, 0, gray.cols, gray.rows)));

    // 放置裁剪图像
    cropped.copyTo(canvas(cv::Rect(original.cols + gray.cols, 0, cropped.cols, cropped.rows)));

    // 在图像上添加文字标签
    cv::putText(canvas, "Original", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "Grayscale", cv::Point(original.cols + 10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "Cropped", cv::Point(original.cols + gray.cols + 10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

    // 显示图像
    cv::imshow("图像处理结果", canvas);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

/**
 * 保存图像文件
 * @param img 要保存的图像
 * @param filepath 保存路径
 * @return 是否保存成功
 */
bool saveImage(const cv::Mat &img, const std::string &filepath)
{
    try
    {
        // 创建输出目录（如果不存在）
        fs::path path(filepath);
        fs::create_directories(path.parent_path());

        bool success = cv::imwrite(filepath, img);
        if (success)
        {
            std::cout << "✅ 图像已保存: " << filepath << std::endl;
        }
        else
        {
            std::cout << "❌ 保存失败: " << filepath << std::endl;
        }
        return success;
    }
    catch (const std::exception &e)
    {
        std::cerr << "保存图像时发生错误: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char **argv)
{
    std::cout << "========================================" << std::endl;
    std::cout << "     OpenCV 图像处理程序" << std::endl;
    std::cout << "========================================" << std::endl;

    // 获取当前工作目录
    std::string currentDir = getCurrentDirectory();
    if (currentDir.empty())
    {
        return -1;
    }

    // 构建图像路径
    std::string imagePath = currentDir + "/images/input/test.jpg";
    std::cout << "\n 读取图片: " << imagePath << std::endl;

    // 检查文件是否存在
    if (!fs::exists(imagePath))
    {
        std::cout << "❌ 错误: 文件不存在 - " << imagePath << std::endl;
        std::cout << "请确保 images/input/ 文件夹中存在 test.jpg 文件" << std::endl;
        return -1;
    }

    // 读取图像
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "❌ 错误: 无法读取图片，请检查文件格式" << std::endl;
        return -1;
    }

    std::cout << "✅ 图片读取成功！" << std::endl;

    // 打印图像信息
    printImageInfo(img, "原始");

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    printImageInfo(gray, "灰度");

    // 裁剪图像左上角区域
    const int CROP_SIZE = 300;
    cv::Mat cropped = cropTopLeft(img, CROP_SIZE);

    // 保存处理后的图像
    std::string outputDir = currentDir + "/images/output";
    fs::create_directories(outputDir);

    std::string grayPath = outputDir + "/grayscale.jpg";
    std::string cropPath = outputDir + "/cropped_top_left.jpg";

    std::cout << "\n 保存图像..." << std::endl;
    saveImage(gray, grayPath);
    saveImage(cropped, cropPath);

    // 显示结果
    std::cout << "\n  按任意键关闭图像窗口..." << std::endl;
    displayImages(img, gray, cropped);

    std::cout << "\n 程序执行完成！" << std::endl;

    return 0;
}
