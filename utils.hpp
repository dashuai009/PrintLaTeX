#pragma once
#include <string>
#include "tokenizer.hpp"
#include <cassert>
#include <torch/torch.h>
#include <random>

namespace utils {

//////// recommend Config1
// model   3048-2
//int d_model = 128;
//int dim_feedforward = 256;
//int n_head = 4;
//float dropout = 0.3;
//int num_decoder_layers = 3;
//int max_output_len = 400;
//// vocabulary
//int token_min_count = 300;

//////// recommend config2
//
int d_model = 256;
int dim_feedforward = 256;
int n_head = 4;
float dropout = 0.1;
int num_decoder_layers = 3;
int max_output_len = 1400;
//
int token_min_count = 2;

torch::Device get_device() {
    auto cuda_available = torch::cuda::cudnn_is_available();
#ifdef DEBUG
    std::cout << "CUDA is " << (cuda_available ? "" : "not ")
              << "available, Running on " << (cuda_available ? "GPU" : "CPU")
              << '\n';
#endif
    torch::Device device{cuda_available ? torch::kCUDA : torch::kCPU};
    return device;
}

int edit_distance(const std::vector<int> &A, const std::vector<int> &B) {

    std::vector<std::vector<int>> dist{{},
                                       {}};
    int pre = 0, cur = 1;
    auto max_len = std::max(A.size(), B.size());
    for (int j = 0; j < max_len + 10; ++j) {
        dist[pre].push_back(j); // 从 空字符串 变换到 B_j
        dist[cur].push_back(0); // 初始化
    }
    for (int i = 1; i <= A.size(); ++i) {
        dist[cur][0] = i;
        for (int j = 1; j <= B.size(); ++j) {
            dist[cur][j] = std::max(i, j); // i直接到j的最大距离是max(i,j)

            dist[cur][j] = std::min(dist[cur][j], dist[pre][j] + 1); // 删除A[i]
            dist[cur][j] =
                    std::min(dist[cur][j], dist[cur][j - 1] + 1); // 删除B[j]
            dist[cur][j] =
                    std::min(dist[cur][j],
                             dist[pre][j - 1] + (A[i - 1] == B[j - 1] ? 0 : 1));
            // A_i != B_j , +1 替换; A_i == B_j , +0;
        }
        std::swap(pre, cur);
    }
    return dist[pre][B.size()];
}

class CharErrorRate {
public:
    explicit CharErrorRate(const std::set<int> &_ignore_indices)
            : ignore_indics(_ignore_indices) {}

    /**
     * 更新错误率
     * @param preds 预测的编码序列 ， (B, )， kInt
     * @param targets 目标的编码序列，(B, )， kInt
     */
    void update(const at::Tensor &preds, const at::Tensor &targets) {
        auto B = preds.size(0);
        assert(preds.scalar_type() == at::kInt);
        assert(targets.scalar_type() == at::kInt);

        for (int64_t i = 0; i < B; ++i) {
            // tensor to cpu
            auto pred_cpu = preds[i].cpu();
            auto tar_cpu = targets[i].cpu();
            std::vector<int> tmp_pred(pred_cpu.data_ptr<int>(),
                                      pred_cpu.data_ptr<int>() +
                                      pred_cpu.numel());
            std::vector<int> tmp_tar(tar_cpu.data_ptr<int>(),
                                     tar_cpu.data_ptr<int>() + tar_cpu.numel());

            // 过滤忽略字符
            std::vector<int> filter_pred, filter_tar;
            for (auto x: tmp_pred) {
                if (ignore_indics.find(x) == ignore_indics.end()) {
                    filter_pred.push_back(x);
                }
            }
            for (auto y: tmp_tar) {
                if (ignore_indics.find(y) == ignore_indics.end()) {
                    filter_tar.push_back(y);
                }
            }
            // 计算编辑距离

            auto dis = edit_distance(filter_pred, filter_tar);
            error += dis * 1.0 / std::max(filter_tar.size(), filter_pred.size());
        }
        total += B;
    }

    double get_error_rate() { return error / total; };

    void clear() {
        error = 0;
        total = 0;
    }

private:
    double error = 0.0;
    int64_t total = 0;
    const std::set<int> ignore_indics;
};

class ProgressBar {
public:
    ProgressBar(const char finish = '#', const char unfini = '.')
            : _flags("-\\/"), _finish(finish), _progress_str(100, unfini),
              _cur_progress(0) {}

    void print_bar(const int n, std::string msg = "") {
        for (int i = _cur_progress; i < n; i++) {
            _progress_str[i] = _finish;
        }
        _cur_progress = n;
        std::string f, p;
        if (n == 100) {
            f = "OK";
            p = "100%";
        } else {
            f = _flags[n % 4];
            p = std::to_string(n) + '%';
        }
        std::cout << std::unitbuf << '[' << f << ']' << '[' << _progress_str
                  << ']' << '[' << p << "] " << msg << '\r';
        if (n >= 100) {
            std::cout << std::endl;
        }
    }

private:
    std::string _flags;
    std::string _progress_str;
    int _cur_progress;
    char _finish;
};

// 随机数相关
std::random_device randomDevice;
std::default_random_engine e1(randomDevice());

bool rand_p(double p0) {
    std::uniform_int_distribution<int> p(0, 100);
    return p0 * 100 > p(e1);//不要等号， p0 == 0 时，不应该随机到
}

double rand_double(double l, double r) {
    std::uniform_real_distribution<double> unif(l, r);
    return unif(e1);
}

} // namespace utils