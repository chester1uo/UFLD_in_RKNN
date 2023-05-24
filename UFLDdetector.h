//
// Created by 罗浩琛 on 2022/8/3.
//

#ifndef RKNN_UFLD_UFLDDETECTOR_H
#define RKNN_UFLD_UFLDDETECTOR_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

#include <vector>

#include "rknn_api.h"
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"

const int tusimple_row_anchor[56] = {64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                                     116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                                     168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                                     220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                                     272, 276, 280, 284};

const int culane_row_anchor[18] = {121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277,
                                   287};
const int custom_anchor[14] = {38, 44, 50, 56, 62, 68,74, 80, 86, 94, 102,115, 128, 143};

const int constant_tusimple_row_anchor[14] = {38*2,  44*2,  50*2,  56*2,  62*2,  68*2,
74*2,  80*2,  86*2,  94*2,  102*2,
115*2,  128*2,   143*2};


struct UFLD_res {

};

class UFLDdetector {

private:
    rknn_context ctx;
    rknn_input_output_num io_info;


    static const int CUSTOM_C = 101;
    static const int CUSTOM_H = 14;
    static const int CUSTOM_W = 2;

    int INPUT_C;
    int INPUT_W;
    int INPUT_H;

    static void dump_tensor_attr(rknn_tensor_attr *attr);

    static unsigned char *load_data(FILE *fp, size_t sz);

    void load_model(const char *filename);

    void *preprocess(cv::Mat src_img);

    static void posrprocess(rknn_output *output, cv::Mat src_img);

public:
    void initDetector(const char *filename);

    void doInference(cv::Mat org_image);

    void destoryDetector();
};


#endif //RKNN_UFLD_UFLDDETECTOR_H
