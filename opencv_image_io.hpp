#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <filesystem>
#include "utils.hpp"

namespace image_io {

void print_img_info(const cv::Mat &img) {
    std::cout << "cols = " << img.cols << " rows = " << img.rows << '\n'
              << "channels = " << img.channels() << " depth = " << (8 << img.depth()) << "bits\n";
}

/**
 * 将tensor转换为8uc3的图片
 * @param tensor 图片的高、宽为tensor的前两维的大小
 * @param mat_type 图片的类型，比如 CV_8UC1
 * @return
 */
auto ToCvImage(const at::Tensor &tensor, int mat_type) -> cv::Mat {
    int height = tensor.sizes()[0];
    int width = tensor.sizes()[1];
    auto img = cv::Mat{cv::Size{width, height}, mat_type, tensor.data_ptr<uchar>()};
    return img.clone();
}

/**
 * 读入图片并一次进行以下处理
 * 1、转灰度
 * 2、仿射变换：x缩放0.6 // 0.5的概率会执行此变换
 * 3、A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
 * 4、A.GaussianBlur(blur_limit=(1, 1), p=0.5),
 * @param file_path 图片路径
 * @return
 */
auto ReadImage_Transform(const std::string &file_path) -> at::Tensor {
    auto img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
//    std::cout << img.rows << ' ' << img.cols << '\n';
    if (utils::rand_p(1)) {// 50%的概率执行缩放
//        std::cout << "scale\n\n" << '\n';
        // 定义仿射变换矩阵
        cv::Mat M = cv::Mat::zeros(2, 3, CV_32FC1);
        float scale_x = utils::rand_double(0.8, 1.2);
        float scale_y = utils::rand_double(0.8, 1.2);
//        std::cout << scale_x << ' ' << scale_y << '\n';
        M.at<float>(0, 0) = scale_x;   // 缩放因子 x
        M.at<float>(1, 1) = scale_y;   // 缩放因子 y
        // 进行仿射变换
        cv::Mat dst = cv::Mat(img.rows * scale_y, img.cols * scale_x , img.type(), cv::Scalar(255));// 灰度图255是白色
        cv::warpAffine(img, dst, M, dst.size());
        img = dst;
    }

    if (utils::rand_p(0.5)) {
        // 设置高斯噪声参数
        double mean = 0.0;  // 噪声均值
        double stddev = utils::rand_double(10, 50);  // 噪声标准差

        // 生成与图像大小相同的高斯噪声
        cv::Mat noise(img.size(), img.type());
        cv::randn(noise, cv::Scalar::all(mean), cv::Scalar::all(stddev));

        // 将噪声添加到图像中
        cv::Mat noisyImage;
        cv::add(img, noise, noisyImage, cv::Mat(), img.type());
        img = noisyImage;
    }

    if (utils::rand_p(0.5)) {
        int ksize = 3;
        double sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
        // https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/blur/functional.py#L36
        // 应用高斯模糊
        cv::Mat dst;
        cv::GaussianBlur(img, dst, cv::Size(ksize, ksize), sigma);
        img = dst;
    }

    // very important!! 注意注意！！这地方必须加clone，不然函数退出时img会被销毁，tensor将会失效
    at::Tensor tensor_image = torch::from_blob(img.data, {img.rows, img.cols, 1}, at::kByte).clone();
    return tensor_image;
}

/**
 * 将8bit 3通道的图片->转为灰度图片->转为tensor
 * @param file_path
 * @return (Rows, Cols, 1) kByte
 */
auto ReadImage_gray(const std::string &file_path) -> at::Tensor {
    if (!std::filesystem::exists(file_path)) {
        std::cout << file_path << " is not exists!\n";
        exit(-2);
    }
    auto img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
//    // Define the scaling factor
//    double scalingFactor = 1.0; // 50% scaling
//
//    // Calculate the new dimensions based on the scaling factor
//    int newWidth = static_cast<int>( img.cols * scalingFactor);
//    int newHeight = static_cast<int>( img.rows * scalingFactor);
//
//    // Resize the image using the new dimensions
//    cv::Mat scaledImage;
//    cv::resize(img, scaledImage, cv::Size(newWidth, newHeight));

    // very important!! 注意注意！！这地方必须加clone，不然函数退出时img会被销毁，tensor将会失效
    at::Tensor tensor_image = torch::from_blob(img.data, {img.rows, img.cols, 1}, at::kByte).clone();
    return tensor_image;
}


namespace test {

void show_image(const cv::Mat &img, const std::string &title) {
    std::string image_type = std::to_string(img.type());
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = {0, 3, 1, 2}) -> at::Tensor {
    std::cout << "############### transpose ############" << std::endl;
    std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    std::cout << "shape after : " << tensor.sizes() << std::endl;
    std::cout << "######################################" << std::endl;
    return tensor;
}


auto TryToCvImage(const at::Tensor &tensor) -> cv::Mat {
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try {
        cv::Mat output_mat = ToCvImage(tensor, CV_8UC3);
        print_img_info(output_mat);
        show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error &e) {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}


} // namesapce test

} // namespace image_io
