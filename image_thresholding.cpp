// Copyright 2018 Zeyu Zhong
// Lincese(MIT)
// Author: Zeyu Zhong
// Date: 2018.5.3

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int image_thresholding() {
    Mat image = imread("���ſ�-��.png", -1);
    Mat image_binary;
    image.copyTo(image_binary);
    std::vector<std::vector<int>> s1(image.rows, std::vector<int>(image.cols, 0));
    //int s1[image.rows][image.cols];
    for (int m = 1; m < image.rows - 1; m++) {
        for (int n = 1; n < image.cols - 1; n++) {
            int D = 0;
            for (int k1 = -1; k1 < 2; k1++) {
                for (int k2 = -1; k2 < 2; k2++) {
                    if (image.at<uchar>(m, n) > image.at<uchar>(m + k1, n + k2))
                        D = (image.at<uchar>(m, n) - image.at<uchar>(m + k1, n + k2) + D);
                    else
                        D = 0 + D;
                }
            }
            s1[m][n] = D;
        }
    }

    int C1 = 0;
    int threshold1 = 0;
    for (int i = 0; i < 256; i++) {
        int S = 0;
        for (int m = 1; m < image.rows - 1; m++) {
            for (int n = 1; n < image.cols - 1; n++) {
                if (image.at<uchar>(m, n) == i)
                    S = S + s1[m][n];
            }
        }
        if (S > C1) {
            C1 = S;
            threshold1 = i;
        }
    }
    for (int m = 1; m < image.rows - 1; m++)
        for (int n = 1; n < image.cols - 1; n++) {
            int D = 0;
            for (int k1 = -1; k1 < 2; k1++)
                for (int k2 = -1; k2 < 2; k2++) {
                    if (image.at<uchar>(m, n) < image.at<uchar>(m + k1, n + k2))
                        D = (image.at<uchar>(m, n) - image.at<uchar>(m + k1, n + k2)) + D;
                }
            s1[m][n] = D;
        }
    int C2;
    int threshold2 = 0;
    for (int i = 0; i < 256; i++) {
        int S = 0;
        for (int m = 1; m < image.rows - 1; m++)
            for (int n = 1; n < image.cols - 1; n++) {
                if (image.at<uchar>(m, n) == i)
                    S = S + s1[m][n];
            }
        if (S != 0 || S > C2) {
            C2 = S;
            threshold2 = i;
        }
    }

    int threshold_final = (threshold2 - threshold1) / 2;

    for (int m = 0; m < image_binary.rows; m++)
        for (int n = 0; n < image_binary.cols; n++) {
            if (image_binary.at<uchar>(m, n) >= threshold_final)
                image_binary.at<uchar>(m, n) = 255;
            else
                image_binary.at<uchar>(m, n) = 0;
        }

    namedWindow("image_binary", WINDOW_AUTOSIZE);
    imshow("image_binary", image_binary);
    //imwrite("../../3-image-thresholding-and-image-refinement/output/image_binary.bmp", image_binary);
    waitKey(0);
    return 0;
}
