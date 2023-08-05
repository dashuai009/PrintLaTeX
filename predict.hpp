#pragma once

#include "resnet_transformer.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <torch/torch.h>


std::vector<std::string> predict(ResNetTransformer model, const Tokenizer &tokenizer, const at::Tensor &input) {
    auto device = utils::get_device();
    torch::NoGradGuard no_grad;
    model->eval();
    model->to(device);// cuda only
    model->train(false);
    auto _x = input.to(at::kFloat).to(device);
    at::Tensor y;
    if (device == at::kCPU) {
        y = (model->predict_cpu(_x)).toType(at::kInt).cpu();
    } else {
        y = (model->predict(_x)).toType(at::kInt).cpu();
    }
    std::vector<std::string> res;
    for (int i = 0; i < y.size(0); ++i) {
        std::vector<int> v(y[i].data_ptr<int>(), y[i].data_ptr<int>() + y[i].numel());
        auto res_str = tokenizer.decode(v);
        res.push_back(res_str);
    }
    return res;
}
