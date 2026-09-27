#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
using namespace std;

class MnistImages {
public:
    int N,rows,cols;
    vector<uint8_t> data;   // 大小 = N*rows*cols，值 0~255
};

vector<uint8_t> load_labels(const string &path);

MnistImages load_images(const string& path);
