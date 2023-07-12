#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <filesystem>

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
    int height= tensor.sizes()[0];
    int width = tensor.sizes()[1];
    auto img = cv::Mat{cv::Size{width, height}, mat_type, tensor.data_ptr<uchar>()};
    return img.clone();
}

/**
 * 将8bit 3通道的图片->转为灰度图片->转为tensor
 * @param file_path
 * @return (Rows, Cols, 1) kByte
 */
auto ReadImage_gray(const std::string &file_path) -> at::Tensor {
    if (!std::filesystem::exists(file_path)){
        std::cout<< file_path << " is not exists!\n";
        exit(-2);
    }
    auto img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    // Define the scaling factor
    double scalingFactor = 0.5; // 50% scaling

    // Calculate the new dimensions based on the scaling factor
    int newWidth = static_cast<int>( img.cols * scalingFactor);
    int newHeight = static_cast<int>( img.rows * scalingFactor);

    // Resize the image using the new dimensions
    cv::Mat scaledImage;
    cv::resize(img, scaledImage, cv::Size(newWidth, newHeight));

    at::Tensor tensor_image = torch::from_blob(scaledImage.data, {scaledImage.rows, scaledImage.cols, 1}, at::kByte).clone();
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
