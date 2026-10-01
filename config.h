#ifndef CONFIG_H_
#define CONFIG_H_
#define SCALAR 10                                     // 暂时没用
#define IMG_NUM 60000                                 // 训练图像数量
#define TRAIN_TIME 30                                 // 训练轮次
#define LEARN_RATE 0.01                               // 学习率
#define LABEL_PATH "src/train-labels.idx1-ubyte"      // 图像标签路径
#define TRAIN_DATA_PATH "src/train-images.idx3-ubyte" // 图像路径
#define WIDTH 28                                      // 图像参数
#define HEIGHT 28
#define HIDDEN_LAYER_NUM 200 // 隐藏层数量
#define TEST_IMG_PATH "src/t10k-images.idx3-ubyte"
#define TEST_LABEL_PATH "src/t10k-labels.idx1-ubyte"
#define AFTER_DATA_PATH "src/result_data.bin"
#define WANT_TO_TRAIN 0
#define WANT_TO_TEST 1
#endif // CONFIG_H_
