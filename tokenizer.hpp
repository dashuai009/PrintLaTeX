#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <set>

class Tokenizer {
public:

    std::vector<int> encode(const std::vector<std::string> &formula) const {
        std::vector<int> res{token_to_index[this->sos_token]};
        for (auto &word: formula) {
            if (token_to_index.find(word) != token_to_index.end()) {
                res.push_back(token_to_index[word]);
            } else {
                res.push_back(token_to_index[unk_token]);
            }
        }
        res.push_back(token_to_index[this->eos_token]);
        return res;
    }

    std::string decode(const std::vector<int> &indices, bool inference = true) const {
        std::string res;
        for (auto index: indices) {
            assert(index_to_token.find(index) != index_to_token.end());
            if (index == token_to_index[eos_token]) {
                break;
            }
            if (inference && (ignore_indices.find(index) != ignore_indices.end())) {
                continue;
            }
            auto token = index_to_token[index];
            if (res.empty()) { res += token; }
            else { res += " " + token; }
        }
        return res;
    }

    Tokenizer()= default;

    /**
     * 从文件中读取字典，该文件与模型对应，应该是>>输出的结果
     * @param file_path
     */
    void load(const std::string &file_path) {
        std::ifstream volcab;
        volcab.open(file_path);
        std::string tok;
        int index;
        while (volcab >> index >> tok) {
            token_to_index[tok] = index;
            index_to_token[index] = tok;
        }
        auto tmp_token = index_to_token[0];
        assert(tmp_token == pad_token);
        auto pad_index = token_to_index[pad_token];
        assert(pad_index == 0);
        for (const auto &token: ignore_tokens) {
            ignore_indices.insert(token_to_index[token]);
        }
    }

    explicit Tokenizer(int min_count) {
        std::ifstream Formulas;
        Formulas.open("data/im2latex_formulas.tok.lst");
        std::string s;
        std::map<std::string, int> word_cnt;
        word_cnt[pad_token] = word_cnt[sos_token] = word_cnt[eos_token] = word_cnt[unk_token] = min_count;

        while (std::getline(Formulas, s)) {
            std::stringstream ss{s};
            std::string word;
            while (ss >> word) {
                word_cnt.try_emplace(word, 0);
                word_cnt[word] += 1;
            }
        }
        std::vector<std::pair<std::string, int>> tmp{word_cnt.begin(), word_cnt.end()};
        std::sort(tmp.begin(), tmp.end());// token各不相同，排序结果唯一，多次读入的index唯一
        int index = 0;
        for (const auto &[token, count]: tmp) {
            if (count >= min_count) {
                token_to_index[token] = index;
                index_to_token[index] = token;
                ++index;
            }
        }

        auto tmp_token = index_to_token[0];
        std::swap(index_to_token[0], index_to_token[token_to_index[pad_token]]);
        std::swap(token_to_index[pad_token], token_to_index[tmp_token]);


        for (const auto &token: ignore_tokens) {
            ignore_indices.insert(token_to_index[token]);
        }
        std::sort(tmp.begin(), tmp.end(), [](const auto &x, const auto &y) {
            return y.second == x.second ? y.first < x.first : x.second < y.second;
        });
        for (auto [token, cnt]: tmp) {
            std::cout << "(" << token << ", " << cnt << ")\n";
        }
    }

    size_t size() {
        return token_to_index.size();
    }

    const std::set<int> get_ignore_indices() {
        return ignore_indices;
    }

    friend std::ostream &operator<<(std::ostream &output,
                                    const Tokenizer &D) {
        for (auto [index, token]: D.index_to_token) {
            output << index << ' ' << token << '\n';
        }
        return output;
    }

    friend std::istream &operator>>(std::istream &input, Tokenizer &D) {
        int index;
        std::string token;
        while (input >> index >> token) {
            D.token_to_index[token] = index;
            D.index_to_token[index] = token;
        }
        return input;
    }

    const std::string pad_token{"<PAD>"}, sos_token{"<SOS>"},
            eos_token{"<EOS>"}, unk_token{"<UNK>"};
private:
    mutable std::map<std::string, int> token_to_index;
    mutable std::map<int, std::string> index_to_token;
    std::set<int> ignore_indices;
    const std::set<std::string> ignore_tokens{pad_token, sos_token, eos_token, unk_token};
};