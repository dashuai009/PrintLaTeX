#pragma once
// #include "data_loader.hpp"
#include "data_loader.hpp"
#include "resnet_transformer.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <torch/torch.h>

void train() {
    auto device = utils::get_device();

//    std::vector<int> milestones{5};
    double lr = 0.0001;
    double weight_decay = 1e-4;
    float gamma = 0.5;
    int num_epochs = 20;
    int batch_size = 16;
    auto optim_scheduler_step = 10;

    utils::PrintLaTeXConfig config;

    auto tokenizer = Tokenizer(2);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[1];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[1];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[1];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[1];

    // training data
    auto train_data_set = LaTeXDataSet::ImageFolderDataset("data", LaTeXDataSet::ImageFolderDataset::Mode::Train);
    auto num_train_samples = train_data_set.size().value();
    std::cout << "train data set = " << num_train_samples << '\n';
    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    std::move(train_data_set), batch_size);

    auto val_cer = utils::CharErrorRate(tokenizer.get_ignore_indices());

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
    model->train();
    model->to(device);

    auto loss_fn = torch::nn::CrossEntropyLoss(
            torch::nn::CrossEntropyLossOptions().ignore_index(pad_index));

    auto optimizer = torch::optim::AdamW(
            model->parameters(),
            torch::optim::AdamWOptions(lr).weight_decay(weight_decay));
    auto scheduler = torch::optim::StepLR(optimizer, optim_scheduler_step, gamma);
    model->eval();

    // model->initWeights();
  torch::load(model, "saved_models/1689620223.pt");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        double running_loss = 0.0;

        utils::ProgressBar pb;
        int cnt = 0;
        for (auto &batch: *train_loader) {
            cnt += batch_size;
            pb.print_bar(cnt * 100 / num_train_samples, "");
            auto [_data, _target] = LaTeXDataSet::collate_fn(batch, tokenizer);
            auto data = _data.to(device);
            auto targets = _target.to(device);
            // std::cout << "_data" << data.sizes() << ' ' << data.scalar_type() << '\n';
            using torch::indexing::Slice, torch::indexing::None;
            auto y = targets.index({Slice(), Slice(None, targets.size(1) - 1)});
            //std::cout << "y = " << y.sizes() << ' ' << y.scalar_type() << '\n';
            auto output = model->forward(data, y);
            //std::cout       << "output = " << output.sizes() << ' ' << output.scalar_type() << '\n'
            //          << "target = " << target.sizes() << ' ' << target.scalar_type() << '\n';
            auto loss = loss_fn(output, targets.index({Slice(), Slice(1, None)}));

            running_loss += loss.item<double>() * data.size(0);

            // validation
            if (cnt >= num_train_samples) {// 最后一个batch
                auto preds = model->predict(data.index({Slice(0, 2)}));
                val_cer.update(preds.toType(at::kInt), targets.index({Slice(0, 2)}).toType(at::kInt), tokenizer);
            }
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            //break;
        }
        auto sample_mean_loss = running_loss / num_train_samples;

        model->eval();
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "],"
                  << "Trainset - Loss: " << sample_mean_loss << ","
                  << "CharErrorRate: " << val_cer.get_error_rate() << '\n';
        if (epoch % 10 == 0) {
            val_cer.clear();
        }
        //break;
    }
    std::cout << "Training finished!\n\n";
//    return;
    auto cur = std::chrono::system_clock::now().time_since_epoch();
    std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(cur);
    torch::save(model, std::string("./saved_models/") + std::to_string(sec.count()) + ".pt");
}