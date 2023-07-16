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
            else { res = res + " " + token; }
        }
        return res;
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

        for (const auto &token: ignore_tokens) {
            ignore_indices.insert(token_to_index[token]);
        }
//        for(auto [index, token]:index_to_token){
//            std::cout << index << '[' << token << "]\n";
//        }
    }

    size_t size() {
        return token_to_index.size();
    }

    const std::set<int> get_ignore_indices() {
        return ignore_indices;
    }

    const std::string pad_token{"<PAD>"}, sos_token{"<SOS>"},
            eos_token{"<EOS>"}, unk_token{"<UNK>"};
private:
    mutable std::map<std::string, int> token_to_index;
    mutable std::map<int, std::string> index_to_token;
    std::set<int> ignore_indices;
    const std::set<std::string> ignore_tokens{pad_token, sos_token, eos_token, unk_token};
};