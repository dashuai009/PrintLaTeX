#pragma once

#include "torch/torch.h"


struct PositionalEncoding1DImpl : torch::nn::Module {
    // https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    explicit PositionalEncoding1DImpl(int d_model, float in_dropout = 0.1, int max_len = 5000) {
        m_dropout = register_module("m_dropout", torch::nn::Dropout(in_dropout));
        m_pe = register_buffer("pe", make_pe(d_model, max_len));
//        std::cout << "m_pe" << m_pe.sizes() << ' ' << m_pe.scalar_type() << '\n';
    }

    static torch::Tensor make_pe(int d_model, int max_len) {
        using torch::indexing::Slice, torch::indexing::None, torch::indexing::Ellipsis;
        torch::Tensor pe = torch::zeros({max_len, d_model});
        auto position = torch::arange(0, max_len, at::ScalarType::Float).unsqueeze(1);
        auto div_term = torch::exp(torch::arange(0, d_model, 2) * (-std::log(1e4) / d_model));
        pe.index_put_({Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
        pe.index_put_({Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));
        return pe.unsqueeze(1);
    }

    /**
     *
     * @param x (S, B, d_model)
     * @return (B, d_model, H, W)
     */
    torch::Tensor forward(const torch::Tensor &x) {
        assert(x.size(2) == m_pe.size(2));
        auto _x = x + m_pe.index({torch::indexing::Slice(torch::indexing::None, x.size(0))});
        return m_dropout(_x);
    }


    torch::nn::Dropout m_dropout = nullptr;
    torch::Tensor m_pe; // (max_len, 1 , d_model)
};

TORCH_MODULE(PositionalEncoding1D);


struct PositionalEncoding2DImpl : torch::nn::Module {
//"""2-D positional encodings for the feature maps produced by the encoder.
//
//Following https://arxiv.org/abs/2103.06450 by Sumeet Singh.
//
//Reference:
//https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/blob/main/lab9/text_recognizer/models/transformer_util.py
//"""
    explicit PositionalEncoding2DImpl(int d_model, int max_h = 2000, int max_w = 2000) : m_d_model(d_model) {
        assert(d_model % 2 == 0);
        m_pe = register_buffer("be", make_pe(d_model, max_h, max_w));
    }

    static at::Tensor make_pe(int d_model, int max_h, int max_w) {
        auto pe_h = PositionalEncoding1DImpl::make_pe(d_model / 2, max_h);
        pe_h = pe_h.permute({2, 0, 1}).expand({-1, -1, max_h});
        auto pe_w = PositionalEncoding1DImpl::make_pe(d_model / 2, max_w);
        pe_w = pe_w.permute({2, 1, 0}).expand({-1, max_h, -1});
        return torch::cat({pe_h, pe_w}, 0);
    }

    at::Tensor forward(const at::Tensor &x) {
        using torch::indexing::Slice, torch::indexing::None, torch::indexing::Ellipsis;

        return x + m_pe.index({Slice(), Slice(None, x.size(2)), Slice(None, x.size(3))});
    }

    int m_d_model;
    torch::Tensor m_pe;
};

TORCH_MODULE(PositionalEncoding2D);