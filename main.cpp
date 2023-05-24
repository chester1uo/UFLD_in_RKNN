#include <iostream>
#include <opencv2/video.hpp>
#include "UFLDdetector.h"

int main(int argc, char **argv) {
    if (argc != 3)
    {
        printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
        return -1;
    }

    char *file_name = (char *)argv[1];
    char *image_name = argv[2];

    UFLDdetector detector;
    detector.initDetector(file_name);
    cv::Mat img = cv::imread(image_name);
    detector.doInference(img);
    return 0;
}
