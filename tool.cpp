#include "tool.h"
#include <fstream>

// 从文件流读 4 字节大端整数
inline uint32_t read_be_u32(ifstream& f) {
    unsigned char buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (uint32_t(buf[0]) << 24) |
           (uint32_t(buf[1]) << 16) |
           (uint32_t(buf[2]) <<  8) |
           (uint32_t(buf[3]));
}
vector<uint8_t> load_labels(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("无法打开标签文件: " + path);

    uint32_t magic = read_be_u32(f);
    if (magic != 0x00000801)
        throw runtime_error("标签文件 magic 错误");

    uint32_t n = read_be_u32(f);

    vector<uint8_t> labels(n);
    f.read(reinterpret_cast<char*>(labels.data()), n);  // 每字节就是标签

    return labels;
}

MnistImages load_images(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("无法打开图像文件: " + path);

    uint32_t magic = read_be_u32(f);
    if (magic != 0x00000803)
        throw runtime_error("图像文件 magic 错误");

    uint32_t N    = read_be_u32(f);
    uint32_t rows = read_be_u32(f);
    uint32_t cols = read_be_u32(f);

    MnistImages img;
    img.N = N; img.rows = rows; img.cols = cols;
    img.data.resize(size_t(N) * rows * cols);
    f.read(reinterpret_cast<char*>(img.data.data()), img.data.size());

    return img;
}
