#pragma once
// #include "data_loader.hpp"
#include "data_loader.hpp"
#include "resnet_transformer.hpp"
#include "tokenizer.hpp"
#include "utils.hpp"
#include "predict.hpp"
#include <torch/torch.h>

void train() {
    auto device = utils::get_device();

    //    std::vector<int> milestones{5};
    double lr = 0.00005;
    double weight_decay = 1e-4;
    float gamma = 0.5;
    int num_epochs = 80;
    int batch_size = 8;
    int num_workers = 0;
    auto optim_scheduler_step = 20;


    auto tokenizer = Tokenizer(utils::token_min_count);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[1];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[1];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[1];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[1];
    std::cout << "volcab size :" << tokenizer.size() << '\n';

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
                    std::move(validate_data_set), 4);

    auto val_cer = utils::CharErrorRate(tokenizer.get_ignore_indices());

    // model
    auto model = ResNetTransformer(
            utils::d_model, utils::dim_feedforward, utils::n_head, utils::dropout,
            utils::num_decoder_layers, utils::max_output_len, sos_index, eos_index,
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

    model->initWeights();
//    torch::load(model, "saved_models/1691228914.pt");

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

            running_loss += loss.item<double>() * data.size(0);
            // 最后一个batch可能不等于batchsize

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
            val_cer.update(preds.toType(at::kInt), targets.toType(at::kInt));
            if (/*epoch != num_epochs - 1 &&*/ cnt_val >= 8) {
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

    std::ofstream vocab_out;
    vocab_out.open(std::string("./saved_models/") + std::to_string(sec.count()) + "_vocab.pt");
    vocab_out << tokenizer;
}


/**
 *
 * @param x (B,
 */
void predict_test() {


    auto tokenizer = Tokenizer(utils::token_min_count);
    auto sos_index = tokenizer.encode({tokenizer.sos_token})[1];
    auto eos_index = tokenizer.encode({tokenizer.eos_token})[1];
    auto pad_index = tokenizer.encode({tokenizer.pad_token})[1];
    auto unk_index = tokenizer.encode({tokenizer.unk_token})[1];

    // model
    auto model = ResNetTransformer(
            utils::d_model,
            utils::dim_feedforward,
            utils::n_head,
            utils::dropout,
            utils::num_decoder_layers,
            utils::max_output_len,
            sos_index,
            eos_index,
            pad_index,
            tokenizer.size()
    );

    std::cout << "asdfsdfg\n";
    auto device = utils::get_device();
        torch::load(model, "saved_models/1689785878.pt", device);
    std::cout << "asdfsdfg\n";
    std::cout << model << '\n';

    std::vector<std::string> test_imgs{"ff5d66560d", "5abbb9b19f", "329a44c373", "73b51f198b", "6e69ea63c3",
                                       "6331d9e7fd", "71b1268d61", "91a55d2cb9", "408fe63a30", "232d6fea7c"};

    auto duration_train = std::chrono::milliseconds{0};
    for (auto img: test_imgs) {
        auto path = std::string("C:\\Users\\15258\\work\\PrintLaTeX\\main\\data\\formula_images_processed\\")
                    + img + ".png";
        auto image = image_io::ReadImage_gray(path);// (h * w * 1)
//        auto img_t = image_io::ToCvImage(image, CV_8UC1);
//        image_io::test::show_image(img_t, "msg");
        auto w = image.size(1);
        auto h = image.size(0);
        auto x = image.reshape({1, 1, h, w}).toType(at::kFloat).div(255); //( 1, 3, H, W)
//        std::cout << "x = " << x.sizes() << ' ' << x.scalar_type() << '\n';

        auto begin_train = std::chrono::system_clock::now();
        auto res = predict(model, tokenizer, x);
        std::cout << img <<'\n';
        for(auto c : res){
            std::cout << c<< ' ';
        }
        std::cout << '\n';
        auto end_train = std::chrono::system_clock::now();
        duration_train += std::chrono::duration_cast<std::chrono::milliseconds>(
                end_train - begin_train);

//
//        std::cout << "len = " << res_str.length() << " res = " << res_str << "\n";
//        int words = 0;
//        for (char c: res_str) {
//            if (c == ' ') {
//                words += 1;
//            }
//        }
//        std::cout << words << '\n';

    }

    std::cout << duration_train.count() * 1.0 / 1000 << "s/10 imgs\n";
}