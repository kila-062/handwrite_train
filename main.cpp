#include "Layer.h"
#include "config.h"
#include "tool.h"
#include <cmath>
#include <cstddef>
#include <fstream>
#include <sstream>

inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// 这个主要起到调试作用，看数据有没有被正确读取
void save_as_ppm(MnistImages src_data) {
    for (int i = 0; i <= 20; i++) {
        ofstream outfile;
        string path;
        ostringstream oss;
        oss << "d:/c/dosomething/dosomething/src/test" << i << ".ppm";
        path = oss.str();

        outfile.open(path, ios::out | ios::binary | ios::trunc);
        char head[32];
        sprintf(head, "P6\n%d %d 255\n", WIDTH, HEIGHT);

        outfile << head;
        for (int s = 0; s < HEIGHT; s++) {

            for (int j = 0; j < WIDTH; j++) {
                int index =
                    s * WIDTH + j +
                    (WIDTH * HEIGHT) * i; // 每个图片像素点的下标 + 第几个图片
                char buffer[3];
                buffer[0] = char(src_data.data[index]);
                buffer[1] = char(src_data.data[index]);
                buffer[2] = char(src_data.data[index]);
                outfile.write(buffer, sizeof(buffer));
            }
        }
        outfile.close();
    }
}
typedef struct {
    unsigned long long data_size;
    char p[0];

} Bin_head;

void save_layer(ofstream &f, Layer &L) {

    f.write((char *)L.get_weight().data(),
            L.get_weight().size() * sizeof(double));
    f.write((char *)L.get_bias().data(), L.get_bias().size() * sizeof(double));
}

void load_layer(ifstream &i, Layer &L) {

    i.read((char *)L.get_weight().data(), L.get_weight_size());

    i.read((char *)L.get_bias().data(), L.get_bias_size());
}

void save_as_bin(Layer layer, Layer hidden) {

    ofstream outfile;
    outfile.open(AFTER_DATA_PATH, ios::out | ios::binary | ios::trunc);

    save_layer(outfile, layer);
    save_layer(outfile, hidden);

    outfile.close();
}

void load_bin_data(Layer &layer, Layer &hidden) {
    ifstream infile;
    infile.open(AFTER_DATA_PATH, ios::binary);

    load_layer(infile, layer);

    load_layer(infile, hidden);
}

// 这个函数封装主要为了服务于目前的3层模型，后面如果再加隐藏层的话不能使用
void forward_pass(Layer &layer, Layer &hidden, const vector<double> &x,
                  vector<double> &h, vector<double> &a) {
    layer.set_input(x);
    layer.feed_forward();
    const auto &z1 = layer.get_output();
    for (int i = 0; i < (int)h.size(); ++i)
        h[i] = sigmoid(z1[i]);

    hidden.set_input(h);
    hidden.feed_forward();
    const auto &z2 = hidden.get_output();
    for (int o = 0; o < (int)a.size(); ++o)
        a[o] = sigmoid(z2[o]);
}
// 计算loss
double compute_loss(const vector<double> &a, int label) {
    double loss = 0;
    for (int o = 0; o < (int)a.size(); ++o) {
        double y = (o == label) ? 1.0 : 0.0;
        loss += -(y * log(a[o] + 1e-9) + (1.0 - y) * log(1.0 - a[o] + 1e-9));
    }
    return loss;
}
double Test_result(Layer &layer, Layer &hidden, const MnistImages &test_images,
                   const vector<double> &all_test_data) {
    double acc = 0;
    auto labels = load_labels(TEST_LABEL_PATH);
    int image_data_index = 0;
    int img_count = 0;
    vector<double> test_data(WIDTH * HEIGHT);
    for (; img_count < test_images.N; img_count++) {

        for (int y = 0; y < HEIGHT * WIDTH; y++, image_data_index++) {

            test_data[y] = all_test_data[image_data_index];
        }

        layer.set_input(test_data);

        layer.feed_forward();

        auto z1 = layer.get_output();

        vector<double> h(layer.get_output_num());

        for (int i = 0; i < (int)h.size(); ++i)
            h[i] = sigmoid(z1[i]);

        hidden.set_input(h);

        hidden.feed_forward();

        int result = hidden.get_result();

        if (result == labels[img_count])
            acc++;
        else if (result != labels[img_count])
            cout << "wrong label = " << static_cast<int>(labels[img_count])
                 << " image count = " << img_count << endl;
    }
    cout << "test_acc = " << acc / test_images.N << endl;
    return acc / test_images.N;
}
void train_pass(MnistImages &images, vector<double> &all_data, Layer &layer,
                Layer &hidden, int epochs, const MnistImages &test_images,
                const vector<double> &all_test_data) {

    auto labels = load_labels(LABEL_PATH);
    cout << "load" << images.N << " pictures " << images.rows << "x"
         << images.cols << "\n";
    // 最大正确率
    int max_acc = 0;

    // 梯度缓冲区，在循环外分配一次
    vector<double> dW2(hidden.get_weight().size(),
                       0.0); // dW2是第二层的权重下降梯度
    vector<double> db2(hidden.get_bias().size(),
                       0.0); // db2是第二层bias的下降梯度
    vector<double> dW1(layer.get_weight().size(), 0.0);
    vector<double> db1(layer.get_bias().size(), 0.0);

    // 中间缓存
    vector<double> h(layer.get_output_num());
    vector<double> a(hidden.get_output_num());
    vector<double> dz2(hidden.get_output_num());
    vector<double> dz1(layer.get_output_num());
    vector<double> train_data(WIDTH * HEIGHT);
    const double lr = LEARN_RATE;

    for (int train_times = 0; train_times < epochs; train_times++) {
        int img_count = 0;
        int image_data_index = 0;
        // double total_loss = 0.0;
        int correct = 0;
        for (; img_count < IMG_NUM; img_count++) {

            for (int y = 0; y < HEIGHT * WIDTH; y++, image_data_index++) {

                train_data[y] = all_data[image_data_index];
            }

            // 让输入层和隐藏层都执行一次feedforward

            forward_pass(layer, hidden, train_data, h, a);

            int label = labels[img_count];

            // 取到总loss,这里可以节省性能省略
            // total_loss += compute_loss(a,label);
            int pred =
                hidden
                    .get_result(); // 这里取到每个图像的结果，目前用的是取概率最大值，暂时先这样（因为用的是sigmoid，所以现在没问题）
            if (pred == label)
                correct++;

            // 反向传播 + 更新bias和weight
            // --- dz2 = a - y  (sigmoid + BCE 的输出层梯度) ---
            for (int o = 0; o < 10; ++o)
                dz2[o] = a[o] - ((o == label) ? 1.0 : 0.0);

            fill(dW2.begin(), dW2.end(), 0.0);
            fill(db2.begin(), db2.end(), 0.0);
            vector<double> dh = hidden.backward(dz2, dW2, db2); // 得到 dL/dh

            // --- dz1 = dh * h * (1 - h)   (sigmoid 导数) ---
            for (int i = 0; i < HIDDEN_LAYER_NUM; ++i)
                dz1[i] = dh[i] * h[i] * (1.0 - h[i]);

            fill(dW1.begin(), dW1.end(), 0.0);
            fill(db1.begin(), db1.end(), 0.0);
            layer.backward(dz1, dW1, db1);

            // 更新两层参数
            layer.apply_gradients(dW1, db1, lr);
            hidden.apply_gradients(dW2, db2, lr);
        }
        int N = IMG_NUM;
        double acc = (double)correct / IMG_NUM;
        cout << "epoch "
             << train_times
             // loss可省略
             //<< "  avg_loss = " << total_loss / IMG_NUM
             << "  train_acc = " << acc;

        double test_acc =
            Test_result(layer, hidden, test_images, all_test_data);

        if (test_acc > max_acc) {

            max_acc = test_acc;
            save_as_bin(layer, hidden);
        }
    }
}
#if WANT_TO_TRAIN
int main() {

    MnistImages images = load_images(TRAIN_DATA_PATH);

    // 训练前提前把数据/255，减少计算
    vector<double> all_data(size_t(images.N) * WIDTH * HEIGHT);
    for (size_t i = 0; i < all_data.size(); ++i)
        all_data[i] = images.data[i] / 255.0;

    MnistImages test_images = load_images(TEST_IMG_PATH);

    vector<double> test_data(size_t(test_images.N) * WIDTH * HEIGHT);
    for (size_t i = 0; i < test_data.size(); ++i)
        test_data[i] = test_images.data[i] / 255.0;
    Layer layer;

    Layer hidden(HIDDEN_LAYER_NUM, 10);

    train_pass(images, all_data, layer, hidden, TRAIN_TIME, test_images,
               test_data);

    system("pause");

    return 0;
}
#endif

#if WANT_TO_TEST
int main() {
    MnistImages test_images = load_images(TEST_IMG_PATH);
    vector<double> test_data(size_t(test_images.N) * WIDTH * HEIGHT);
    for (size_t i = 0; i < test_data.size(); ++i)
        test_data[i] = test_images.data[i] / 255.0;
    Layer layer;
    Layer hidden(HIDDEN_LAYER_NUM, 10);
    load_bin_data(layer, hidden);
    double acc = Test_result(layer, hidden, test_images, test_data);

    system("pause");
    return 0;
}
#endif
