#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "PictureMatch.h"
#include <math.h>
#include <numeric>

cv::Scalar PictureMatch::getMSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

MatchBox PictureMatch::getmatchbox(cv::Mat& targetImage, cv::Mat& templateImage) {
    // 进行模板匹配
    cv::Mat result; //这是匹配结果图片
    cv::matchTemplate(targetImage, templateImage, result, cv::TM_CCOEFF_NORMED);

    // 寻找最佳匹配位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);  

    cv::Mat img0 = templateImage.clone();
    cv::Mat img1 = targetImage(cv::Rect(maxLoc.x, maxLoc.y, templateImage.cols, templateImage.rows));
    return   MatchBox(maxLoc.x, maxLoc.y, maxLoc.x + templateImage.cols, maxLoc.y + templateImage.rows, (float)maxVal, pixelmatching(img0, img1));
}

std::vector<MatchBox> PictureMatch::getmatchboxs(cv::Mat& image, cv::Mat& templateImage,double inthreshold) {
    // 准备输出结果
    cv::Mat result;
    result.create(image.cols - templateImage.cols + 1, image.rows - templateImage.rows + 1, CV_32F);

    // 匹配模板
    cv::matchTemplate(image, templateImage, result, cv::TM_CCOEFF_NORMED);

    // 查找最佳匹配位置
    std::vector<MatchBox> locations;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    // 多处匹配
    double threshold = inthreshold * maxVal; // 阈值系数，可以根据需要调整
    while (true) {
        // 找到大于阈值的位置
        if (maxVal <= threshold) {
            break;
        }
        cv::Mat img0 = templateImage.clone();
        cv::Mat img1 = image(cv::Rect(maxLoc.x, maxLoc.y, templateImage.cols, templateImage.rows));
        locations.push_back(MatchBox(maxLoc.x, maxLoc.y, maxLoc.x + templateImage.cols, maxLoc.y + templateImage.rows, (float)maxVal, pixelmatching(img0, img1)));

        // 移动搜索窗口，继续匹配
        cv::floodFill(result, maxLoc, cv::Scalar(0));
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    }
    return locations;
}

void PictureMatch::MAT2WBC(cv::Mat& image, cv::Mat& edges,bool boostborder) {
    // 转换成灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
    if (boostborder)
    {
        // 可选：使用形态学操作加强边界
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(edges, edges, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);
    }
}


float PictureMatch::pixelmatching(cv::Mat& image1, cv::Mat& image2, int Failover) {
    int all = image1.rows * image1.cols;
    int cout = 0;
    // 转换为灰度图
    cv::Mat gray1;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    // 转换为灰度图
    cv::Mat gray2;
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    // 遍历图片中的每个像素
    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            // 获取两张图片在(x, y)位置的像素值
            uchar pixel1 = gray1.at<uchar>(y, x);
            uchar pixel2 = gray2.at<uchar>(y, x);
            // 对比像素值，这里只做BGR分量的简单对比
            if (abs((int)pixel1- (int)pixel2) <= Failover) {
                cout++;
            }
        }
    }
    return   (float)cout / (float)all;
}

cv::Mat PictureMatch::dealwithblue(cv::Mat& image) {
    // 创建一个与原图相同大小的掩膜，并用0填充
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    // 遍历图片中的每个像素
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取(x, y)位置的像素值
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            // 如果像素不是是蓝色的，将掩膜中的对应位置设为255
            if ((uchar)color[2] < (uchar)color[0] || (uchar)color[1]  > (uchar)color[2]) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    // 使用掩膜提取蓝色的像素
    cv::Mat bluePixels;
    image.copyTo(bluePixels, mask);
    return bluePixels;
}



float PictureMatch::BrightnessEqualizationContrast(cv::Mat& img1, cv::Mat& img2) {
    // 计算两张图片的直方图均衡化 必须是单通道
    cv::Mat histogramEqualizedImg1, histogramEqualizedImg2;
    cv::equalizeHist(img1, histogramEqualizedImg1);
    cv::equalizeHist(img2, histogramEqualizedImg2);
    cv::imshow("histogramEqualizedImg1", histogramEqualizedImg1);
    cv::imshow("histogramEqualizedImg2", histogramEqualizedImg2);
    cv::waitKey(0);
    int all = img1.rows * img1.cols;
    int cout = 0;
    std::vector<double> data;
    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            // 获取两张图片在(x, y)位置的像素值
            uchar pixel1 = histogramEqualizedImg1.at<uchar>(y, x);
            uchar pixel2 = histogramEqualizedImg2.at<uchar>(y, x);
            // 对比像素值，这里只做BGR分量的简单对比
            if (abs((int)pixel1 - (int)pixel2)<=20) {
                cout++;
            }      
        }
    }
    return  (float)cout/(float)all;
}

float PictureMatch::SingleChannelPixelComparison(cv::Mat& img1, cv::Mat& img2, RBGCHANNEL channel, int gap) {
    int cout = 0;

    // 遍历图片中的每个像素
    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img2.cols; ++x) {
            // 获取两张图片在(x, y)位置的像素值
            cv::Vec3b pixel1 = img1.at<cv::Vec3b>(y, x);
            cv::Vec3b pixel2 = img2.at<cv::Vec3b>(y, x);
            if (abs(pixel1[(size_t)channel] - pixel2[(size_t)channel]) <= 20) {
                cout++;
            }
        }
    }
    return (float)cout / (float)(img1.rows * img1.cols);
}

float binaryimagepixelcmp(cv::Mat& image1, cv::Mat& image2) {
    int all = image1.rows * image1.cols;
    int cout = 0;

    // 遍历图片中的每个像素
    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            // 获取两张图片在(x, y)位置的像素值
            uchar pixel1 = image1.at<uchar>(y, x);
            uchar pixel2 = image2.at<uchar>(y, x);
            // 对比像素值，这里只做BGR分量的简单对比
            if (pixel1 == pixel2) {
                cout++;
            }
        }
    }
    return   (float)cout / (float)all;
}

float PictureMatch::BGR2GRAYCMP(cv::Mat& img1, cv::Mat& img2,double SplitValue) {
    cv::Mat gray1;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::Mat gray2;
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    double thresh1 = getgraylightestthresh(gray1, SplitValue);
    double thresh2 = getgraylightestthresh(gray2, SplitValue);
    cv::Mat edge1;
    cv::threshold(gray1, edge1, thresh1, 255, cv::THRESH_BINARY); 
    cv::Mat edge2;
    cv::threshold(gray2, edge2, thresh2, 255, cv::THRESH_BINARY);
    //cv::imshow("gray1", gray1);
    //cv::imshow("gray2", gray2);
    //cv::imshow("edge1", edge1);
    //cv::imshow("edge2", edge2);
    //cv::waitKey(0);
    return  binaryimagepixelcmp(edge1, edge2);
}

double PictureMatch::getgraylightestthresh(cv::Mat& image, double thresh) {
    int t = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取(i, j)位置的像素值
            uchar pixelValue = image.at<uchar>(y, x);
            t += (int)pixelValue;
        }
    }
    float autothresh = t / (image.rows * image.cols) * thresh;
    if (autothresh > 255)
        autothresh = 255;
    return autothresh;
}

double PictureMatch::getgraythreshLDR(cv::Mat& image, uchar threshdif) {
    int once = 0;
    uchar t_thresh = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取(i, j)位置的像素值
            uchar pixelValue = image.at<uchar>(y, x);
            if (once == 0)
            {
                once = 1;
                t_thresh = pixelValue;
            }
            if (abs((int)pixelValue - t_thresh)<=10)
            {

            }
            else
            {

            }
        }
    }
}