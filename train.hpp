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
    int num_workers = 4;
    auto optim_scheduler_step = 10;

    utils::PrintLaTeXConfig config;

    auto tokenizer = Tokenizer(2);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[1];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[1];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[1];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[1];

    // training data
    auto train_data_set = LaTeXDataSet::ImageFolderDataset(
            "data", LaTeXDataSet::ImageFolderDataset::Mode::Train);
    auto num_train_samples = train_data_set.size().value();
    std::cout << "train data set = " << num_train_samples << '\n';
    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    std::move(train_data_set), torch::data::DataLoaderOptions(batch_size).workers(num_workers));

    // validate data
    auto validate_data_set = LaTeXDataSet::ImageFolderDataset(
            "data", LaTeXDataSet::ImageFolderDataset::Mode::Validate);
    auto num_validate_samples = validate_data_set.size().value();
    std::cout << "validate data set = " << num_validate_samples << '\n';
    auto validate_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    std::move(validate_data_set), batch_size / 4);

    auto val_cer = utils::CharErrorRate(tokenizer.get_ignore_indices());

    // model
    auto model = ResNetTransformer(
            config.d_model, config.dim_feedforward, config.n_head, config.dropout,
            config.num_decoder_layers, config.max_output_len, sos_index, eos_index,
            pad_index, tokenizer.size());
    model->train();
    model->to(device);

    auto loss_fn = torch::nn::CrossEntropyLoss(
            torch::nn::CrossEntropyLossOptions().ignore_index(pad_index));

    auto optimizer = torch::optim::AdamW(
            model->parameters(),
            torch::optim::AdamWOptions(lr).weight_decay(weight_decay));
    auto scheduler =
            torch::optim::StepLR(optimizer, optim_scheduler_step, gamma);
    model->eval();

    // model->initWeights();
    torch::load(model, "saved_models/1689785878.pt");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        double running_loss = 0.0;

        utils::ProgressBar pb;
        int cnt = 0;
        auto begin_train = std::chrono::system_clock::now();
        for (auto &batch: *train_loader) {
            cnt += batch.size();
            pb.print_bar(cnt * 100 / num_train_samples, "");
            auto [_data, _target] = LaTeXDataSet::collate_fn(batch, tokenizer);
            auto data = _data.to(device);
            auto targets = _target.to(device);
            // std::cout << "_data" << data.sizes() << ' ' << data.scalar_type()
            // << '\n';
            using torch::indexing::Slice, torch::indexing::None;
            auto y = targets.index({Slice(), Slice(None, targets.size(1) - 1)});
            // std::cout << "y = " << y.sizes() << ' ' << y.scalar_type() <<
            // '\n';
            auto output = model->forward(data, y);
            // std::cout       << "output = " << output.sizes() << ' ' <<
            // output.scalar_type() << '\n'
            //           << "target = " << target.sizes() << ' ' <<
            //           target.scalar_type() << '\n';
            auto loss =
                    loss_fn(output, targets.index({Slice(), Slice(1, None)}));

            running_loss += loss.item<double>() *
                            data.size(0); // 最后一个batch可能不等于batchsize

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            //    break;
        }
        auto end_train = std::chrono::system_clock::now();
        auto duration_train = std::chrono::duration_cast<std::chrono::seconds>(
                end_train - begin_train)
                .count();
        auto sample_mean_loss = running_loss / num_train_samples;

        utils::ProgressBar pb_val;
        int cnt_val = 0;
        auto begin_validate = std::chrono::system_clock::now();
        for (auto &batch: *validate_loader) {
            cnt_val += batch.size();
            pb_val.print_bar(cnt_val * 100 / num_validate_samples, "");
            auto [_data, _target] = LaTeXDataSet::collate_fn(batch, tokenizer);
            auto data = _data.to(device);
            auto targets = _target.to(device);
            auto preds = model->predict(data);
            val_cer.update(preds.toType(at::kInt), targets.toType(at::kInt), tokenizer);
            if (epoch != num_epochs - 1 && cnt_val >= 300) {
                pb_val.print_bar(100, "");
                break; // 每4个epoch才会完全测试一遍validate数据集
            }
        }
        auto end_validate = std::chrono::system_clock::now();
        auto duration_validate =
                std::chrono::duration_cast<std::chrono::seconds>(
                        end_validate - begin_validate).count();
        model->eval();
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "],"
                  << "Trainset - Loss: " << sample_mean_loss << ","
                  << "CharErrorRate: " << val_cer.get_error_rate()
                  << "trainning duration: " << duration_train << "s "
                  << "validate duration: " << duration_validate << "s \n";
        val_cer.clear();
    }
    std::cout << "Training finished!\n\n";
    //    return;
    auto cur = std::chrono::system_clock::now().time_since_epoch();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(cur);
    torch::save(model, std::string("./saved_models/") +
                       std::to_string(sec.count()) + ".pt");

}