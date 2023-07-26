#pragma once
// #include "data_loader.hpp"
#include "data_loader.hpp"
#include "resnet_transformer.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <torch/torch.h>


std::vector<std::string> predict(ResNetTransformer model, const Tokenizer &tokenizer, const at::Tensor &input) {
    auto device = utils::get_device();
    torch::NoGradGuard no_grad;
    model->eval();
    model->to(device, at::kHalf);// cuda only
    model->train(false);
    auto _x = input.to(at::kHalf).to(device);
    auto y = (model->predict(_x)).toType(at::kInt).cpu();
    // std::cout << y << '\n';
    std::vector<std::string> res;
    std::cout << "y = " << y.sizes() << ' ' << y.scalar_type() << '\n';
    for (int i = 0; i < y.size(0); ++i) {
        std::cout << "y[i]" << y[i].sizes() << '\n';
        // std::cout << "output" << y[i] << '\n';
        std::vector<int> v(y[i].data_ptr<int>(), y[i].data_ptr<int>() + y[i].numel());
        auto res_str = tokenizer.decode(v);
        res.push_back(res_str);
        std::cout << "len = " << res_str.length() << " res = " << res_str << "\n";
        int words = 0;
        for (char c: res_str) {
            if (c == ' ') {
                words += 1;
            }
        }
        std::cout << words << '\n';
    }
    return res;
}

/**
 *
 * @param x (B,
 */
void predict_test() {

    utils::PrintLaTeXConfig config;


    auto tokenizer = Tokenizer(2);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[1];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[1];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[1];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[1];

    // model
    // model
    auto model = ResNetTransformer(
            config.d_model,
            config.dim_feedforward,
            config.n_head,
            config.dropout,
            config.num_decoder_layers,
            config.max_output_len,
            sos_index,
            eos_index,
            pad_index,
            tokenizer.size()
    );

    torch::load(model, "saved_models/1690387944.pt");
    std::cout << model << '\n';


    auto image = image_io::ReadImage_gray(
            "C:\\Users\\15258\\work\\PrintLaTeX\\main\\data\\formula_images_processed\\ff5d66560d.png");// (h * w * 1)
    auto img_t = image_io::ToCvImage(image, CV_8UC1);
    image_io::test::show_image(img_t, "msg");
    auto w = image.size(1);
    auto h = image.size(0);
    auto x = image.reshape({1, 1, h, w}).toType(at::kFloat).div(255); //( 1, 3, H, W)
    std::cout << "x = " << x.sizes() << ' ' << x.scalar_type() << '\n';

    predict(model, tokenizer, x);
}