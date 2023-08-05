//
// Created by dashuai009 on 2023/6/2.
//
#pragma once

// #include "stb_image_io.hpp"
#include "opencv_image_io.hpp"
#include "tokenizer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <torch/torch.h>
#include <vector>

#include <assert.h>

namespace fs = std::filesystem;

namespace LaTeXDataSet {

/**
 * first: 1 * h * w, channel == 1的灰度图
 * second: formula string
 */
using LaTeXDataType = std::pair<at::Tensor, std::vector<std::string>>;

std::vector<std::string> Split(const std::string &s) {
    std::vector<std::string> res;
    std::stringstream ss(s);
    std::string word;
    while (ss >> word) {
        res.push_back(word);
        if (res.size() > utils::max_output_len - 10) {//过长的公式直接截断
            break;
        }
    }
    assert(res.size() < utils::max_output_len);
    return res;
}

/**
 * Dataset class that provides image-label samples.
 */
class ImageFolderDataset
        : public torch::data::datasets::Dataset<ImageFolderDataset, LaTeXDataType> {
public:
    enum class Mode {
        Train, TinyTrain, Validate, Test
    };

    explicit ImageFolderDataset(const std::string &root,
                                Mode mode = Mode::TinyTrain,
                                torch::IntArrayRef image_load_size = {})
            : data_root(root), mode_(mode),
              image_load_size_(image_load_size.begin(), image_load_size.end()) {
        std::ifstream F, D;
        F.open(root + "/im2latex_formulas.tok.lst");
        std::string buf;
        while (std::getline(F, buf)) {
            Formulas.push_back(buf);
        }
        if (mode == Mode::Train || mode == Mode::TinyTrain) {
            D.open(root + "/im2latex_train.lst");
        } else if (mode == Mode::Test) {
            D.open(root + "/im2latex_test.lst");
        } else if (mode == Mode::Validate) {
            D.open(root + "/im2latex_train.lst");
        }
        int index;
        std::string image_name, tmp;
        while (D >> index >> image_name >> tmp) {
            samples_.emplace_back(image_name, Formulas[index]);
            if ((mode == Mode::TinyTrain || mode == Mode::Validate) && samples_.size() == 200) {
                // 小训练集，500个
                break;
            }
        }
    }

    LaTeXDataType get(size_t index) override {
        const auto &[image_name, formula] = samples_[index];
        auto file_path =
                data_root + "/formula_images_processed/" + image_name + ".png";

        auto image = mode_ == Mode::Validate
                     ? image_io::ReadImage_gray(file_path)
                     : image_io::ReadImage_Transform(file_path);
        // (h * w * 1)
        auto w = image.size(1);
        auto h = image.size(0);
        return {image.reshape({1, h, w}), Split(formula)};
    }

    [[nodiscard]] torch::optional<size_t> size() const override {
        return samples_.size();
    }

private:
    std::string data_root;
    Mode mode_;
    std::vector<int64_t> image_load_size_;
    std::vector<std::string> classes_;
    std::vector<std::pair<std::string, std::string>>
            samples_; // (image_name, formula)
    std::vector<std::string> Formulas;
};

/**
 * 1、预处理一个batch的图片，将所有图片大小调整到总共最大的宽高，随机横竖平移图片，不足的位置补0
 * 2、对齐formula的长度
 * @param batch 一个batch的数据，若干个(1*h*w的灰度图, 公式)
 * @param tokenizer
 * @return { (B, 1, MaxH, MaxW) kFloat, (B, MaxLen) kLong }
 */
std::pair<at::Tensor, at::Tensor>
collate_fn(const std::vector<LaTeXDataType> &batch,
           const Tokenizer &tokenizer) {
    auto batch_size = static_cast<int64_t>(batch.size());
    auto max_H = batch[0].first.size(1);
    auto max_W = batch[0].first.size(2);
    auto max_len = batch[0].second.size();
    for (auto &[_image, _formulas]: batch) {
        max_H = std::max(max_H, _image.size(1));
        max_W = std::max(max_W, _image.size(2));
        max_len = std::max(max_len, _formulas.size());
    }
    auto padded_images = torch::full({batch_size, 1, max_H, max_W}, 255).toType(at::kByte);
    auto batched_indices =
            torch::full({batch_size, static_cast<int64_t>(max_len) + 2}, 0)
                    .toType(at::ScalarType::Long); // 0 <PAD>

    std::random_device r;
    std::default_random_engine e1(r());
    int i = 0;
    for (auto &[_image, _formula]: batch) {
        using torch::indexing::Slice, torch::indexing::None,
                torch::indexing::Ellipsis;
        auto H = _image.size(1);
        auto W = _image.size(2);
        std::uniform_int_distribution<int> rand_H(0, max_H - H);
        auto y = rand_H(e1);
        std::uniform_int_distribution<int> rand_W(0, max_W - W);
        auto x = rand_W(e1);
        padded_images.index_put_({i, Slice(), Slice(y, y + H), Slice(x, x + W)},
                                 _image);
        auto indices = tokenizer.encode(_formula);
        batched_indices.index_put_(
                {i, Slice(None, indices.size())},
                torch::from_blob(indices.data(),
                                 {static_cast<long long>(indices.size())},
                                 torch::TensorOptions(at::kInt)));
        i += 1;
    }
    return {padded_images.to(at::kFloat).div(255).clone(), batched_indices};
}

} // namespace LaTeXDataSet