#include "Layer.h"
#include <random>

Layer::Layer(int i, int o)
    : input(i, 0.0), output(o, 0.0), weight(i * o, 0.0), bias(o, 0.0),
      input_num(i), output_num(o) {
    // Xavier 初始化：标准差 sqrt(1 / input_num)
    static mt19937 gen(42); // 固定种子，结果可复现
    normal_distribution<double> dist(0.0, sqrt(1.0 / i));
    for (auto &w : weight)
        w = dist(gen);
    for (auto &b : bias)
        b = 0.0; // bias 可以保持 0
}

void Layer::feed_forward() {

    // ±éÀúÃ¿Ò»¸öÒþ²Ø²ãÉñ¾­Ôª
    for (int z = 0; z < output_num; ++z) {
        double sum = 0.0f;

        // ±éÀúÕ¹Æ½ºóµÄËùÓÐÊäÈëÏñËØ
        for (int i = 0; i < input_num; ++i) {
            sum += input[i] * weight[z * input_num + i];
        }
        sum += bias[z];
        // ½«¼ÓÈ¨ºÍ¸³Öµ¸øÒþ²Ø²ãÉñ¾­Ôª£¨´Ë´¦¿É¼Ó¼¤»îº¯Êý£¬Èç ReLU/Sigmoid£©
        output[z] = sum;
    }
}

void Layer::set_input(const vector<double> &new_input) { input = new_input; }
const vector<double> &Layer::get_output() const { return output; }

const vector<double> &Layer::get_weight() const { return weight; }

const vector<double> &Layer::get_bias() const { return bias; }

const int &Layer::get_output_num() const { return output_num; }

const int &Layer::get_input_num() const { return input_num; }

size_t Layer::get_weight_size() const {
    auto weight_bytesize = weight.size() * sizeof(double);
    return weight_bytesize;
}

size_t Layer::get_bias_size() const {
    auto bias_bytesize = bias.size() * sizeof(double);
    return bias_bytesize;
}

int Layer::get_result() const {
    if (output.empty()) {
        return -1;
    }
    double max = output[0];
    int result = 0;
    for (int i = 1; i < output.size(); i++) {
        if (output[i] > max) {
            max = output[i];
            result = i;
        }
    }
    return result;
}

vector<double> Layer::backward(const vector<double> &dz, vector<double> &dW,
                               vector<double> &db) const {
    vector<double> dx(input_num, 0.0);

    for (int o = 0; o < output_num; ++o) {
        double g = dz[o];
        double *gw = &dW[o * input_num];
        const double *wrow = &weight[o * input_num];

        for (int i = 0; i < input_num; ++i) {
            gw[i] += g * input[i]; // dL/dW[o,i] = dz[o] * x[i]
            dx[i] += g * wrow[i];  // dL/dx[i] = ¦² dz[o] * W[o,i]
        }
        db[o] += g; // dL/db[o] = dz[o]
    }
    return dx;
}

void Layer::apply_gradients(const vector<double> &dW, const vector<double> &db,
                            double lr) {
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] -= lr * dW[i];
    for (size_t o = 0; o < bias.size(); ++o)
        bias[o] -= lr * db[o];
}
