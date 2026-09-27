#pragma once
#include <vector>
#include <iostream>
#include <cstdint>
#define WIDTH 28
#define HEIGHT 28
#define HIDDEN_LAYER_NUM 15

using namespace std;
class Layer {
public:

	Layer(int input_num = WIDTH*HEIGHT,int output_num = HIDDEN_LAYER_NUM);

	void feed_forward();

	const vector<double>& get_output() const;

    int get_result() const;

    const vector<double> &get_weight() const;

    const vector<double> &get_bias() const;

    const int &get_output_num()const;

    const int &get_input_num() const;

    void set_input(const vector<double>& new_input);

	vector<double> backward(const vector<double>& dz,
                               vector<double>& dW,
                               vector<double>& db) const;//dW是权重，db是bias

    void apply_gradients(const vector<double>& dW,
                            const vector<double>& db,
                            double lr);//dW是权重，db是bias，lr是学习率，用来更新权重，bias和学习率
private:
	vector<double>input;

	vector<double>output;

	vector<double>weight;

    vector<double> bias;

	int input_num;

	int output_num;



};
