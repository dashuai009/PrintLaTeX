#pragma once
// #include "data_loader.hpp"
#include "data_loader.hpp"
#include "resnet_transformer.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <torch/torch.h>

/**
 *
 * @param x (B,
 */
void predict() {
    auto device = utils::get_device();

    utils::PrintLaTeXConfig config;


    auto tokenizer = Tokenizer(2);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[0];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[0];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[0];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[0];

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

//    torch::load(model, "saved_models/1689175011.pt");
    std::cout << model << '\n';

    torch::NoGradGuard no_grad;
    model->eval();
    model->to(device);
    model->initWeights();
    model->train(false);

    auto test_data_set = LaTeXDataSet::ImageFolderDataset("data");
    std::cout << "train data set = " <<  test_data_set.size().value() << '\n';
    auto test_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    std::move(test_data_set), 1);

    for (auto &batch: *test_loader) {
        auto [_data, _target] = LaTeXDataSet::collate_fn(batch, tokenizer);
        auto data = _data.to(device);
        std::cout << data.sizes() << '\n';
        auto target = _target.to(device);

        auto y = (model->predict(data)).toType(at::kInt);
        std::cout << y << '\n';
        break;
    }

    auto image = image_io::ReadImage_gray("C:\\Users\\15258\\work\\PrintLaTeX\\main\\data\\formula_images\\1a50fa207d.png");// (h * w * 1)

    auto w = image.size(1);
    auto h = image.size(0);
    auto x = image.reshape({1, 1, h, w}).toType(at::kFloat); //( 1, 3, H, W)
    std::cout << "x = " << x.sizes() << ' ' << x.scalar_type() << '\n';
//
    auto _x = x.to(device);
    auto y = (model->predict(_x)).toType(at::kInt).cpu();
//    std::cout << y << '\n';
    std::cout << "y = " << y.sizes() << ' ' << y.scalar_type() << '\n';
    for (int i = 0; i < y.size(0); ++i) {
        std::cout << "output" << y[i] << '\n';
        std::vector<int> v(y[i].data_ptr<int>(), y[i].data_ptr<int>() + y[i].numel());
        auto res = tokenizer.decode(v);
        std::cout << res << "\n\n";
    }
}