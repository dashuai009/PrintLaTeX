//
// Created by dashuai009 on 7/11/2023.
//

#include <chrono>
#include <iostream>

std::string currentDateTime() {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);

    char buffer[128];
    strftime(buffer, sizeof(buffer), "%m-%d-%Y %X", now);
    return buffer;
}

int main(){
    auto cur  = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(cur);
    std::cout << "Current Time and Date: " << std::ctime(&end_time) << std::endl;

    return 0;
}