#pragma once
#include <torch/torch.h>


namespace utils {

struct PrintLaTeXConfig{
    // model args
    int d_model = 128;
    int dim_feedforward = 256;
    int n_head = 4;
    float dropout = 0.3;
    int num_decoder_layers = 3;
    int max_output_len = 1500;
};


torch::Device get_device() {

    auto cuda_available = torch::cuda::cudnn_is_available();
    std::cout << "CUDA is " << (cuda_available ? "" : "not ") << "avaiable, Training on "
              << (cuda_available ? "GPU" : "CPU") << '\n';
    torch::Device device{cuda_available ? torch::kCUDA : torch::kCPU};
    return device;
}
} // namespace utils