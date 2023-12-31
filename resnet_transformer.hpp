#pragma once

#include <utility>

#include "positional_encoding.hpp"
#include "resnet/impl.hpp"

using at::Tensor;

Tensor generate_square_subsequent_mask(int in_size) {
    auto mask = torch::triu(torch::ones({in_size, in_size}) == 1).transpose(0, 1);
    mask = mask
            .to(torch::kFloat32)
            .masked_fill(mask == 0, -std::numeric_limits<float>::infinity())
            .masked_fill(mask == 1, float(0.0));
    return mask;
}

/**
 *  Find the first occurence of element in x along a given dimension.
 *
    Usage:
        >>> first_element(Tensor([[1, 2, 3], [2, 3, 3], [1, 1, 1]]), 3)
        tensor([2, 1, 3])

    Reference:
        https://discuss.pytorch.org/t/first-nonzero-index/24769/9
        https://github.com/kingyiusuen/image-to-latex/blob/main/image_to_latex/models/resnet_transformer.py#L176

        I fixed an edge case where the element we are looking for is at index 0. The
        original algorithm will return the length of x instead of 0.
    """
 * @tparam T
 * @param x  The input tensor to be searched.
 * @param element The number to look for.
 * @param dim  The dimension to reduce.
 * @return  Indices of the first occurence of the element in x. If not found, return the length of x along dim.
 */
template<typename T = int>
Tensor find_first(const Tensor &x, int element, int dim = 1) {
    auto mask = (x == element);
    auto [found, indices] = ((mask.cumsum(dim) == 1) & mask).max(dim);
    indices.index_put_({(~found) & (indices == 0)}, x.size(dim));
    return indices;
}

class ResNetTransformerImpl : public torch::nn::Module {
public:
    ResNetTransformerImpl() = delete;

    ResNetTransformerImpl(
            int d_model,
            int dim_feedforward,
            int n_head,
            float dropout,
            int num_decoder_layers,
            int max_output_len,
            int sos_index,
            int eos_index,
            int pad_index,
            int num_classes)
            : d_model(d_model),
              max_output_len(max_output_len + 2),
              sos_index(sos_index),
              eos_index(eos_index),
              pad_index(pad_index) {
        _resnet = register_module("resnet", resnet_space::resnet18(10));
//        backbone = torch::nn::Sequential{
//                _resnet->m_conv1,
//                _resnet->m_bn1,
//                _resnet->m_relu,
//                _resnet->m_maxpool
//        };
        bottleneck = register_module("bottleneck", torch::nn::Conv2d(256, d_model, 1));
        image_position_encoder = register_module("image_position_encoder", PositionalEncoding2D(d_model));

        // decoder
        embedding = register_module("embedding", torch::nn::Embedding(num_classes, d_model));
        y_mask = register_buffer("y_mask", generate_square_subsequent_mask(this->max_output_len));
        word_position_encoder = register_module(
                "word_position_encoder",
                PositionalEncoding1D(d_model, /*in_dropout =*/ dropout, this->max_output_len)
        );
        transformer_decoder = register_module("transformer_decoder", torch::nn::TransformerDecoder(
                torch::nn::TransformerDecoderLayer(
                        torch::nn::TransformerDecoderLayerOptions(d_model, n_head)
                                .dim_feedforward(dim_feedforward)
                                .dropout(dropout)
                ),
                num_decoder_layers
        ));
        fc = register_module("fc", torch::nn::Linear(d_model, num_classes));
    }

    void initWeights() {
        auto init_range = 0.1;
        embedding->weight.data().uniform_(-init_range, init_range);
        fc->bias.data().zero_();
        fc->weight.data().uniform_(-init_range, init_range);

        torch::nn::init::kaiming_normal_(
                bottleneck->weight.data(),
                0,
                torch::kFanOut,
                torch::kReLU
        );

        auto [x, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(
                bottleneck->weight.data()
        );
        auto bound = 1.0 / std::sqrt(fan_out);
        torch::nn::init::normal_(bottleneck->bias, -bound, bound);
    }

    /**
     *
     * @param x (B, _E, _H, _W)
     * @param y (B, Sy) with elements in (0, num_classes - 1)
     * @return (B, num_classes, Sy) logits
     */
    Tensor forward(const Tensor &x, Tensor y) {
        auto encoded_x = encode(x);  // (Sx, B, E)
        //std::cout << "encoded_x: " << encoded_x.sizes() << '\n';
        auto output = decode(std::move(y), encoded_x); //# (Sy, B, num_classes)
        //std::cout << output.sizes() << ' ' << output.scalar_type() << '\n';
        return output.permute({1, 2, 0}); // # (B, num_classes, Sy)
    }

    /**
     *
     * @param x: (Batch_size, Chanel, _H, _W)
     * @return
     */
    Tensor encode(Tensor x) {
        if (x.size(1) == 1) {
            x = x.repeat({1, 3, 1, 1});
        }
        // std::cout << "to be encoded = " << x.sizes() << ' ' << x.scalar_type() << '\n';
        // resnet
//        x = backbone->forward(x);
        x = _resnet->m_conv1->forward(x);
        x = _resnet->m_bn1->forward(x);
        x = _resnet->m_relu->forward(x);
        x = _resnet->m_maxpool->forward(x);
        x = _resnet->m_layer1->forward(x);
        x = _resnet->m_layer2->forward(x);
        x = _resnet->m_layer3->forward(x);
        // end resnet

        x = bottleneck->forward(x);
        x = image_position_encoder->forward(x);
        x = x.flatten(2).permute({2, 0, 1});
        return x;
    }

    /**
     * Decode encoded inputs with teacher-forcing.
     * @param y (B, Sy) with elements in (0, num_classes - 1)
     * @param x (Sx, B, E)
     * @return   (Sy, B, num_classes) logits
     */
    Tensor decode(Tensor y, const Tensor &x) {
        using torch::indexing::Slice, torch::indexing::None, torch::indexing::Ellipsis;

        //std::cout << " input y" << y.sizes() << ' ' << y.scalar_type() << '\n';
        y = y.permute({1, 0}); // (Sy, B)
        //std::cout << y.sizes() << ' ' << y.scalar_type() << '\n';
        y = embedding(y) * std::sqrt(d_model); // (Sy, B, E)
        //std::cout << "after embedding" << y.sizes() << ' ' << y.scalar_type() << '\n';
        y = word_position_encoder->forward(y); // (Sy, B, E)
        auto sy = y.size(0);
        auto _y_mask = y_mask.index({Slice(None, sy), Slice(None, sy)})
                .type_as(x);
        auto output = transformer_decoder->forward(y, x, _y_mask);
        return fc(output);
    }

    /**
     * Make predictions at inference time.
     * @param x :Input images, (B, C, H, W), at::kFloat
     * @return (B, max_output_len) with elements in (0, num_classes - 1). at::kFloat
     */
    Tensor predict(const Tensor &x) {
//        std::cout << "predict" << x.scalar_type() << ' ' << x.sizes() << '\n';
        using namespace torch::indexing;
        auto B = x.size(0);
        auto S = max_output_len;

        auto encoded_x = encode(x); // (Sx, B, E)

        auto output_indices = torch::full({B, S}, pad_index).type_as(x).toType(at::kLong);
        output_indices.index_put_({Slice(), 0}, sos_index);
        auto has_ended = torch::full({B,}, false);

        for (int sy = 1; sy < S; ++sy) {
            auto y = output_indices.index({Slice(), Slice(None, sy)}); // (B, sy)
            auto logits = decode(y, encoded_x);
            auto output = torch::argmax(logits, -1);// (sy, B)
            output_indices.index_put_({Slice(), sy}, output.index({Slice(output.size(0) - 1, None)}));
            has_ended |= (output_indices.index({Slice(), sy}) == eos_index).type_as(has_ended);
            if (torch::all(has_ended).item<bool>()) {
                break;
            }
        }

        auto eos_positions = find_first(output_indices, eos_index);
        for (int i = 0; i < B; ++i) {
            int j = int(eos_positions[i].item<int>()) + 1;
            output_indices.index_put_({i, Slice(j, None)}, pad_index);
        }
//        std::cout << output_indices.index({Slice(), Slice(0, 20)}) << '\n';
        return output_indices;
    }

    /**
 * Make predictions at inference time.
 * @param x : Input image, (1, C, H, W), at::kFloat
 * @return (max_output_len) with elements in (0, num_classes - 1). at::kFloat
 */
    Tensor predict_cpu(const Tensor &x) {
        using namespace torch::indexing;
        auto B = x.size(0);
        assert(B == 1);
        auto S = max_output_len;

        auto encoded_x = encode(x); // (Sx, 1, E)

        auto output_indices = torch::full({B, S}, pad_index, at::TensorOptions(at::kInt));
        output_indices[0][0] = sos_index;

        for (int sy = 1; sy < S; ++sy) {
            auto y = output_indices.index({Slice(), Slice(None, sy)});
            // output_indices[0][sy]; // (B = 1, sy)
            auto logits = decode(y, encoded_x);
            auto output = torch::argmax(logits, -1);// (sy, B = 1)
            output_indices.index_put_({Slice(), sy}, output.index({Slice(output.size(0) - 1, None)}));
            if (output_indices[0][sy].item<int>() == eos_index) {
                break;
            }
        }
//        std::cout << output_indices.index({Slice(), Slice(0, 20)}) << '\n';
        return output_indices;
    }


private:
    int d_model, max_output_len, sos_index, eos_index, pad_index;
//    torch::nn::Conv2d backone;


    ResNet<BasicBlock> _resnet{nullptr};
    //torch::nn::Sequential backbone{nullptr};
    torch::nn::Conv2d bottleneck{nullptr};
    PositionalEncoding2D image_position_encoder = nullptr;

    torch::nn::Embedding embedding = nullptr;
    Tensor y_mask;
    PositionalEncoding1D word_position_encoder = nullptr;
    torch::nn::TransformerDecoder transformer_decoder = nullptr;
    torch::nn::Linear fc = nullptr;

};

TORCH_MODULE(ResNetTransformer);