//
// Created by 罗浩琛 on 2022/8/3.
//

#include <csignal>
#include "UFLDdetector.h"


double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }


static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float) qnt - (float) zp) * scale;
}


unsigned char *UFLDdetector::load_data(FILE *fp, size_t sz) {
    unsigned char *data = nullptr;
    int ret;

    ret = fseek(fp, 0, SEEK_SET);
    if (ret != 0) {
        printf("Blob seek failure.\n");
        return nullptr;
    }

    data = (unsigned char *) malloc(sz);
    if (data == nullptr) {
        printf("Buffer malloc failure.\n");
        return nullptr;
    }
    fread(data, 1, sz, fp);
    return data;
}

void UFLDdetector::load_model(const char *filename) {
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (nullptr == fp) {
        printf("Open file %s failed.\n", filename);
        return;
    }

    fseek(fp, 0, SEEK_END);
    int size = (int) ftell(fp);
    printf("Successfully open file %s, with file size %d\n", filename, size);
    data = load_data(fp, size);

    fclose(fp);
    int ret = rknn_init(&ctx, data, size, 0, nullptr);
    if (ret < 0) {
        printf("RKNN init error ret=%d\n", ret);
        return;
    }
}


void *UFLDdetector::preprocess(cv::Mat src_img) {
    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    cv::Mat img;
    cv::cvtColor(src_img, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;
    printf("Input img width = %d, Input img height = %d\n", img_width, img_height);
    auto *resize_buf = new float[INPUT_W * INPUT_H * INPUT_C];

    src = wrapbuffer_virtualaddr((void *) img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *) resize_buf, INPUT_W, INPUT_H, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS) ret));
        return nullptr;
    }
    IM_STATUS STATUS = imresize(src, dst);
    cv::Mat resize_img(cv::Size(INPUT_W, INPUT_H), CV_32FC3, resize_buf);
    return resize_buf;
}


void UFLDdetector::initDetector(const char *filename) {
    load_model(filename);
    int ret = 0;
    if (ret < 0) {
        printf("RKNN init error ret=%d\n", ret);
        return;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("RKNN init error ret=%d\n", ret);
        return;
    }
    printf("SDK version: %s Driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("RKNN init error ret=%d\n", ret);
        return;
    }
    io_info = io_num;
    printf("Model input num: %d, Output num: %d\n", io_num.n_input, io_num.n_output);

    // Input and output parameters settings
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("Rknn_init error ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));

    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("Model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        height = input_attrs[0].dims[3];
    } else {
        printf("Model is NHWC input fmt\n");
        width = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    INPUT_H = width;
    INPUT_W = height;
    INPUT_C = channel;
    printf("Model input height=%d, width=%d, channel=%d\n", width, height, channel);


}


void UFLDdetector::doInference(cv::Mat org_image) {

    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    cv::Mat img;
    cv::cvtColor(org_image, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;
    printf("Input img width = %d, Input img height = %d\n", img_width, img_height);

    void *resize_buf = malloc(INPUT_W * INPUT_H * INPUT_C);

    src = wrapbuffer_virtualaddr((void *) img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *) resize_buf, INPUT_W, INPUT_H, RK_FORMAT_RGB_888);

    int ret = imcheck(src, dst, src_rect, dst_rect);

    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS) ret));
        //return nullptr;
    }

    IM_STATUS STATUS = imresize(src, dst);
    cv::Mat resize_img(cv::Size(INPUT_W, INPUT_H), CV_8UC3, resize_buf);
    cv::imwrite("resize.jpg", resize_img);

    rknn_input inputs[1];

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = INPUT_H * INPUT_W * INPUT_C;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resize_buf;


    rknn_inputs_set(ctx, io_info.n_input, inputs);
    rknn_output outputs[io_info.n_output];
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;
    memset(outputs, 0, sizeof(outputs));

    struct timeval start_time{}, stop_time{};
    gettimeofday(&start_time, nullptr);
    rknn_run(ctx, nullptr);
    rknn_outputs_get(ctx, io_info.n_output, outputs, nullptr);
    gettimeofday(&stop_time, nullptr);
    printf("Inference time: %f ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    posrprocess(&outputs[0], org_image);

}

void UFLDdetector::destoryDetector() {
    rknn_destroy(this->ctx);
}

void UFLDdetector::posrprocess(rknn_output *output, cv::Mat src_img) {
    auto *ch = (float16_t *) output[0].buf;
    float output_tensor[CUSTOM_C][CUSTOM_H][CUSTOM_W];
    float softmax_tensor[CUSTOM_C - 1][CUSTOM_H][CUSTOM_W];
    float location[CUSTOM_H][CUSTOM_W];
    float colmax[CUSTOM_H][CUSTOM_W];

    cv::Mat out_img = src_img.clone();
    int LAYER_SIZE = CUSTOM_H * CUSTOM_W;
    // Create output tensor
    for (int c = 0; c < CUSTOM_C; c++) {
        for (int h = 0; h < CUSTOM_H; h++) {
            for (int w = 0; w < CUSTOM_W; w++) {
                int idx = c * LAYER_SIZE + h * CUSTOM_W + w;
                float16_t temp = ch[idx];
                output_tensor[c][h][w] = temp;
            }
        }
    }


    // Sofrmax
    for (int h = 0; h < CUSTOM_H; h++){
        for (int w = 0; w < CUSTOM_W; w++)  {
            float sum = 0;
            for (int c = 0; c < CUSTOM_C - 1; c++) {
                sum += exp(output_tensor[c][h][w]);
            }
            for (int c = 0; c < CUSTOM_C - 1; c++) {
                softmax_tensor[c][h][w] = exp(output_tensor[c][h][w]) / sum;
            }
        }
    }

    for (int h = 0; h < CUSTOM_H; h++) {
        for (int w = 0; w < CUSTOM_W; w++) {
            float sum = 0;
            float max = -999999.0;
            float max_loc = 0;
            for (int c = 0; c < CUSTOM_C - 1; c++) {
                if (softmax_tensor[c][h][w] > max) {
                    max = softmax_tensor[c][h][w];
                    max_loc = c;
                }
                sum += softmax_tensor[c][h][w] * (c);
            }
            location[h][w] = max_loc;

            if (max_loc == CUSTOM_W - 1) {
                colmax[h][w] = 0;
            } else {
                colmax[h][w] = sum;
            }
        }
    }

    for (int h = 0; h < CUSTOM_H; h++) {
        for (int w = 0; w < CUSTOM_W; w++) {
            if (colmax[h][w] > 1) {
                int x = (int) colmax[h][w] * 4 * src_img.cols / 400 - 1;
                int y = (int) src_img.rows * custom_anchor[h] / 144 - 1;
                cv::Point pp = {x, y};
                cv::circle(out_img, pp, 5, cv::Scalar(255, 0, 255));
            }
        }
    }

    cv::imwrite("result.jpg", out_img);

}

void UFLDdetector::dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);

}






