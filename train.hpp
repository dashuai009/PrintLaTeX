#pragma once
// #include "data_loader.hpp"
#include "data_loader.hpp"
#include "resnet_transformer.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include <torch/torch.h>


class ProgressBar {
public:
    ProgressBar(const char finish = '#', const char unfini = '.')
            : _flags("-\\/"),
              _finish(finish),
              _progress_str(100, unfini),
              _cur_progress(0) {}

    void print_bar(const ushort n) {
//        if (_cur_progress != 0 && n <= _cur_progress) {
//            std::cerr << "\e[31merror\e[m: n(" << n
//                      << ") should > _cur_progress("
//                      << _cur_progress << ")" << std::endl;
//            return ;
//        }
        for (ushort i = _cur_progress; i < n; i++) {
            _progress_str[i] = _finish;
        }
        _cur_progress = n;
        std::string f, p;
        if (n == 100) {
            f = "\e[1;32mOK\e[m";
            p = "\e[1;32m100%\e[m";
        } else {
            f = _flags[n % 4];
            p = std::to_string(n) + '%';
        }
        std::cout << std::unitbuf
                  << '[' << f << ']'
                  << '[' << _progress_str << ']'
                  << '[' << p << "]" << '\r';
        if (n >= 100) {
            std::cout << std::endl;
        }
    }

private:
    std::string _flags;
    std::string _progress_str;
    ushort _cur_progress;
    char _finish;
};

void train() {
    auto device = utils::get_device();

    double lr = 0.001;
    double weight_decay = 1e-4;

    std::vector<int> milestones{5};
    float gamma = 0.1;

    utils::PrintLaTeXConfig config;

    int num_epochs = 15;
    int batch_size = 16;

    auto tokenizer = Tokenizer(2);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[0];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[0];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[0];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[0];

    // training data
    auto train_data_set = LaTeXDataSet::ImageFolderDataset("data");
    auto num_train_samples = train_data_set.size().value();
    std::cout << "train data set = " << num_train_samples << '\n';
    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                    std::move(train_data_set), batch_size);

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
    model->eval();

    model->initWeights();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "epoch: " << epoch << '\n';
        model->train();
        double running_loss = 0.0;
        size_t num_correct = 0;

        ProgressBar pb;
        int cnt = 0;
        for (auto &batch: *train_loader) {
            cnt += batch_size;
            pb.print_bar(cnt * 100 / num_train_samples);
            auto [_data, _target] = LaTeXDataSet::collate_fn(batch, tokenizer);
            auto data = _data.to(device);
            auto target = _target.to(device);
            // std::cout << "_data" << data.sizes() << ' ' << data.scalar_type() << '\n';
            using torch::indexing::Slice, torch::indexing::None;
            auto y = target.index({Slice(), Slice(None, target.size(1) - 1)});
            //std::cout << "y = " << y.sizes() << ' ' << y.scalar_type() << '\n';
            auto output = model(data, y);
            //std::cout       << "output = " << output.sizes() << ' ' << output.scalar_type() << '\n'
            //          << "target = " << target.sizes() << ' ' << target.scalar_type() << '\n';
            auto loss = loss_fn(output, target.index({Slice(), Slice(1, None)}));
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

        model->eval();
        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "],"
                  << "Trainset - Loss: " << sample_mean_loss << ","
                  << "Accuracy: " << accuracy << '\n';
    }
    std::cout << "Training finished!\n\n";
    auto cur = std::chrono::system_clock::now().time_since_epoch();
    std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(cur);
    torch::save(model, std::string("./saved_models/") + std::to_string(sec.count()) + ".pt");
}