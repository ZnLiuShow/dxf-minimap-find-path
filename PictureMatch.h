#pragma once

class MatchBox
{
public:
    MatchBox() {};
    MatchBox(int X1, int Y1, int X2, int Y2, float sc,float psc) {
        x1 = X1; y1 = Y1; x2 = X2; y2 = Y2; score = sc; pixsc = psc;
    }
    int x1;
    int y1;
    int x2;
    int y2;
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };
    float score;
    float pixsc;
    float area() { return getWidth() * getHeight(); };
};
enum class RBGCHANNEL
{
    red = 0,
    green=1,
    blue=2
};
class PictureMatch
{
public:
	cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2);
    MatchBox getmatchbox(cv::Mat& targetImage, cv::Mat& templateImage);
    std::vector<MatchBox> getmatchboxs(cv::Mat& targetImage, cv::Mat& templateImage,double inthreshold = 0.9);
    void MAT2WBC(cv::Mat& image, cv::Mat& edges,bool boostborder = false);//增加识别的成功率
    float pixelmatching(cv::Mat& image1, cv::Mat& image2, int Failover = 10);
    cv::Mat dealwithblue(cv::Mat& image);
    float BrightnessEqualizationContrast(cv::Mat& img1, cv::Mat& img2);
    float SingleChannelPixelComparison(cv::Mat& img1,cv::Mat&img2, RBGCHANNEL channel,int gap=20);
    float BGR2GRAYCMP(cv::Mat& img1, cv::Mat& img2, double SplitValue);
private:
    double getgraylightestthresh(cv::Mat& image, double thresh);
    double getgraythreshLDR(cv::Mat& image, uchar threshdif);//LDR算法取最佳阈值
};

