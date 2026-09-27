#include "Layer.h"
#include "tool.h"
#include <sstream>
#include <fstream>
#include <cmath>
#define SCALAR 10
#define IMG_NUM 10
inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
void save_as_ppm(MnistImages src_data) {
    for(int i = 0;i<=20;i++){
        ofstream outfile;
        string path;
        ostringstream oss;
        oss << "d:/c/dosomething/dosomething/src/test" << i << ".ppm";
        path = oss.str();


        outfile.open(path,ios::out|ios::binary|ios::trunc);
        char head[32];
        sprintf(head,"P6\n%d %d 255\n",WIDTH,HEIGHT);
        //outfile << "P6\n"  << WIDTH << " " << HEIGHT << endl << "255" <<endl;
        outfile << head;
        for (int s = 0; s < HEIGHT; s++) {


            for (int j = 0; j < WIDTH; j++) {
                int index = s*WIDTH + j + (WIDTH*HEIGHT)*i;//每个图片像素点的下标 + 第几个图片
                char buffer[3];
                buffer[0] = char(src_data.data[index]);
                buffer[1] = char(src_data.data[index]);
                buffer[2] = char(src_data.data[index]);
                outfile.write(buffer,sizeof(buffer));
            }
        }
        outfile.close();
    }
}

void save_as_bin(vector<double> data) {


}

void train_pass(MnistImages images, Layer &layer, Layer &hidden){

    auto labels = load_labels("d:/c/dosomething/dosomething/src/train-labels.idx1-ubyte");
    cout << "load" << images.N << " pictures "
                  << images.rows << "x" << images.cols << "\n";

    //梯度缓冲区，在循环外分配一次
    vector<double> dW2(hidden.get_weight().size(),0.0);//dW2是第二层的权重下降梯度
    vector<double> db2(hidden.get_bias().size(),0.0);//db2是第二层bias的下降梯度
    vector<double> dW1(layer.get_weight().size(), 0.0);
    vector<double> db1(layer.get_bias().size(), 0.0);


      // 中间缓存
    vector<double> h(layer.get_output_num());
    vector<double> a(hidden.get_output_num());
    vector<double> dz2(hidden.get_output_num());
    vector<double> dz1(layer.get_output_num());

    const double lr = 0.1;
    int img_count = 0;
    int image_data_index = 0;
    for(;img_count<IMG_NUM;img_count++){
        vector<double> train_data(WIDTH * HEIGHT);
        for (int y = 0; y < HEIGHT*WIDTH; y++,image_data_index++) {

            train_data[y] = images.data[image_data_index]/255.0;//数据需要除以255.0

        }


        // ============ 4. 训练前的预测 ============
        auto forward = [&]() {
            layer.set_input(train_data);
            layer.feed_forward();
            const auto& z1 = layer.get_output();
            for (int i = 0; i <h.size(); ++i) h[i] = sigmoid(z1[i]);

            hidden.set_input(h);
            hidden.feed_forward();
            const auto& z2 = hidden.get_output();
            for (int o = 0; o < 10; ++o) a[o] = sigmoid(z2[o]);
        };

        int label   = labels[img_count];
        auto compute_loss = [&]() {
            double loss = 0;
            for (int o = 0; o < a.size(); ++o) {
                double y = (o == label) ? 1.0 : 0.0;
                loss += -(y * log(a[o] + 1e-9) + (1.0 - y) * std::
                          log(1.0 - a[o] + 1e-9));
            }
            return loss;
        };

        forward();
        cout << "before: label=" << label
             << "  loss=" << compute_loss() << "\n";


        // ============ 5. 反向 + 更新 ============
        // --- dz2 = a - y  (sigmoid + BCE 的输出层梯度) ---
        for (int o = 0; o < 10; ++o)
            dz2[o] = a[o] - ((o == label) ? 1.0 : 0.0);

        std::fill(dW2.begin(), dW2.end(), 0.0);
        std::fill(db2.begin(), db2.end(), 0.0);
        vector<double> dh = hidden.backward(dz2, dW2, db2);   // 得到 dL/dh

        // --- dz1 = dh * h * (1 - h)   (sigmoid 导数) ---
        for (int i = 0; i < 15; ++i)
            dz1[i] = dh[i] * h[i] * (1.0 - h[i]);

        std::fill(dW1.begin(), dW1.end(), 0.0);
        std::fill(db1.begin(), db1.end(), 0.0);
        layer.backward(dz1, dW1, db1);

        // --- 更新两层参数 ---
        layer.apply_gradients(dW1, db1, lr);
        hidden.apply_gradients(dW2, db2, lr);

        // ============ 6. 训练后的预测 ============
        forward();
        cout << "after : label=" << label << "  loss=" << compute_loss()
             << "\n";

        cout << "trained" << img_count << "times" << endl;
    }

}
int main()
{

    MnistImages images = load_images("d:/c/dosomething/dosomething/src/train-images.idx3-ubyte");

    Layer layer;

    Layer hidden(15,10);

    train_pass(images,layer,hidden);

    system("pause");
    return 0;
}
