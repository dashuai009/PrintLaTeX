#pragma once
#include<string>


class ImageToLatex {
    ImageToLatex(int batch_size = 8, int num_workers = 0, bool pin_memory = false) :
            batch_size(batch_size), num_workers(num_workers), pin_memory(pin_memory) {

    }

    void setup(){

    }

private:
    int batch_size, num_workers;
    bool pin_memory;

    std::string data_dir;
};