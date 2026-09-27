#include "Layer.h"
#include "tool.h"
#include <sstream>
#include <fstream>
#include <cmath>
#define SCALAR 10
#define IMG_NUM 60000
#define TRAIN_TIME 30
#define LABEL_PATH "d:/c/dosomething/dosomething/src/train-labels.idx1-ubyte"
#define TRAIN_DATA_PATH "d:/c/dosomething/dosomething/src/train-images.idx3-ubyte"
inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }




//这个主要起到调试作用，看数据有没有被正确读取
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
    //之后写，目前思路是将用到图层的weight和bias打包，需要有一个头来表示图层数量，数据大小，下面的body直接写入数据
}

//这个函数封装主要为了服务于目前的3层模型，后面如果再加隐藏层的话不能使用
void forward_pass(Layer& layer, Layer& hidden,
                  const vector<double>& x,
                  vector<double>& h,
                  vector<double>& a)
{
    layer.set_input(x);
    layer.feed_forward();
    const auto& z1 = layer.get_output();
    for (int i = 0; i < (int)h.size(); ++i)
        h[i] = sigmoid(z1[i]);

    hidden.set_input(h);
    hidden.feed_forward();
    const auto& z2 = hidden.get_output();
    for (int o = 0; o < (int)a.size(); ++o)
        a[o] = sigmoid(z2[o]);
}
double compute_loss(const vector<double>& a, int label) {
    double loss = 0;
    for (int o = 0; o < (int)a.size(); ++o) {
        double y = (o == label) ? 1.0 : 0.0;
        loss += -(y * log(a[o] + 1e-9)
                + (1.0 - y) * log(1.0 - a[o] + 1e-9));
    }
    return loss;
}

void train_pass(MnistImages images,vector<double> &all_data ,Layer &layer, Layer &hidden,int epochs){

    auto labels = load_labels(LABEL_PATH);
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
    vector<double> train_data(WIDTH * HEIGHT);
    const double lr = 0.1;

    for (int train_times = 0; train_times < epochs; train_times++) {
        int img_count = 0;
        int image_data_index = 0;
        //double total_loss = 0.0;
        int correct = 0;
        for(;img_count<IMG_NUM;img_count++){

            for (int y = 0; y < HEIGHT*WIDTH; y++,image_data_index++) {

                train_data[y] = all_data[image_data_index];

            }

            // 让输入层和隐藏层都执行一次feedforward

            forward_pass(layer, hidden, train_data,h,a);

            int label   = labels[img_count];


            //取到总loss,这里可以节省性能省略
            //total_loss += compute_loss(a,label);
            int pred = hidden.get_result();//这里取到每个图像的结果，目前用的是取概率最大值，暂时先这样（因为用的是sigmoid，所以现在没问题）
            if (pred == label)
                correct++;


            // 反向传播 + 更新bias和weight
            // --- dz2 = a - y  (sigmoid + BCE 的输出层梯度) ---
            for (int o = 0; o < 10; ++o)
                dz2[o] = a[o] - ((o == label) ? 1.0 : 0.0);

            fill(dW2.begin(), dW2.end(), 0.0);
            fill(db2.begin(), db2.end(), 0.0);
            vector<double> dh = hidden.backward(dz2, dW2, db2);   // 得到 dL/dh

            // --- dz1 = dh * h * (1 - h)   (sigmoid 导数) ---
            for (int i = 0; i < 15; ++i)
                dz1[i] = dh[i] * h[i] * (1.0 - h[i]);

            fill(dW1.begin(), dW1.end(), 0.0);
            fill(db1.begin(), db1.end(), 0.0);
            layer.backward(dz1, dW1, db1);

            //更新两层参数
            layer.apply_gradients(dW1, db1, lr);
            hidden.apply_gradients(dW2, db2, lr);

            }
        int N = IMG_NUM;
        double acc = (double)correct / IMG_NUM;
        cout << "epoch "
             << train_times
            //loss可省略
            //<< "  avg_loss = " << total_loss / IMG_NUM
             << "  train_acc = " << acc << endl;

    }
}
int main()
{

    MnistImages images = load_images(TRAIN_DATA_PATH);

    // 训练前提前把数据/255，减少计算
    vector<double> all_data(size_t(images.N) * WIDTH * HEIGHT);
    for (size_t i = 0; i < all_data.size(); ++i)
        all_data[i] = images.data[i] / 255.0;


    Layer layer;

    Layer hidden(15,10);

    train_pass(images,all_data,layer,hidden,TRAIN_TIME);

    system("pause");

    return 0;
}
