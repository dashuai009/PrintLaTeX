#pragma once
// #include "data_loader.hpp"
#include "../data_loader.hpp"
#include "../resnet_transformer.hpp"
#include "../tokenizer.hpp"
#include "../utils.hpp"
#include <torch/torch.h>

int main() {
    auto device = utils::get_device();

    double lr = 0.001;
    double weight_decay = 1e-4;

    std::vector<int> milestones{5};
    float gamma = 0.1;

    int d_model = 128;
    int dim_feedforward = 256;
    int n_head = 4;
    float dropout = 0.3;
    int num_decoder_layers = 3;
    int max_output_len = 150;

    int num_epochs = 1;
    int batch_size = 1;

    auto tokenizer = Tokenizer(2);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[0];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[0];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[0];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[0];

    auto train_data_set = LaTeXDataSet::ImageFolderDataset("data");
    auto num_train_samples = train_data_set.size().value();
    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    std::move(train_data_set), batch_size);
    auto model = ResNetTransformer(
            d_model, dim_feedforward, n_head, dropout, num_decoder_layers,
            max_output_len, sos_index, eos_index, pad_index, tokenizer.size());
    auto loss_fn = torch::nn::CrossEntropyLoss(
            torch::nn::CrossEntropyLossOptions().ignore_index(pad_index));

    auto optimizer = torch::optim::AdamW(
            model->parameters(),
            torch::optim::AdamWOptions(lr).weight_decay(weight_decay));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto &batch: *train_loader) {
            for (auto &it: batch[0].second) {
                std::cout << it << " " ;
            }
            std::cout << batch[0].first.sizes() << '\n';
            auto img1 = image_io::ToCvImage(
                    batch[0].first.clone().reshape({
                                                           batch[0].first.size(1), batch[0].first.size(2), 1
                                                   }),
                    CV_8UC1);
            image_io::test::show_image(img1, "img1");

            auto [_data, _target] = LaTeXDataSet::collate_fn(batch, tokenizer);
            std::cout << _data.sizes() << '\n';
            auto img_sample = _data[0].reshape({_data[0].size(1), _data[0].size(2), 1});
            auto img = image_io::ToCvImage(img_sample, CV_8UC1);
            image_io::test::show_image(img, "img1");
            std::cout << _target[0] <<'\n';
            break;
            auto data = _data.to(device);
            auto target = _target.to(device);
            using torch::indexing::Slice, torch::indexing::None;
            std::cout << data.sizes() << ' ' << target.sizes() << '\n';
            auto output =
                    model->forward(data, target.index({Slice(), Slice(None, _target.size(1))}));

            auto loss =
                    loss_fn(output, target.index({Slice(), Slice(1, None)}));
//            running_loss += loss.item<double>() * data.size(0);
//
//            auto prediction = output.argmax(1);
//
//            num_correct += prediction.eq(target).sum().item<int64_t>();

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "],"
                  << "Trainset - Loss: " << sample_mean_loss << ","
                  << "Accuracy: " << accuracy << '\n';
    }
    std::cout << "Training finished!\n\n";
//    torch::save(model, "./saved_models");
    return 0;
}