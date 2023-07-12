//
// Created by dashuai009 on 7/2/2023.
//
#include <random>
#include <iostream>

int main() {
    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> rand_H(0, 10);

    for (int i = 1; i <= 10; ++i) {
        auto y = rand_H(e1);
        std::cout << y << '\n';
    }
    return 0;
}