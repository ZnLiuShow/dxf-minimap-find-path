// opencv_mp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/img_hash.hpp>
#include <iostream>
#include "PictureMatch.h"

//double calculateSimilarity(const cv::Point& p1, const cv::Point& p2) {
//    // 计算两点之间的欧氏距离
//    int distance = cv::norm(cv::Point2f(p1.x - p2.x, p1.y - p2.y));
//
//    // 标准化距离，得到相似度（注意：这里的最大距离应该是图像的宽度或高度的最大值）
//    double similarity = 1.0 - (distance / std::max(p1.x, p2.x));
//
//    return similarity;
//}
//
//
//float angle(cv::Point pt1, cv::Point pt0, cv::Point pt2)//角度计算
//{
//    double dx1 = (pt1.x - pt0.x);
//    double dy1 = (pt1.y - pt0.y);
//    double dx2 = (pt2.x - pt0.x);
//    double dy2 = (pt2.y - pt0.y);
//    double angle_line = (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
//    double a = acos(angle_line) * 180 / 3.141592653;
//    return a;
//}

uchar getgraylightest(cv::Mat& image) {
    int t = 0;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取(i, j)位置的像素值
            uchar pixelValue = image.at<uchar>(y, x);
            t += (int)pixelValue;
        }      
    }
    return (uchar)(t/(image.rows* image.cols));
}
void My_BGR2GRAY(cv::Mat& img1, cv::Mat& img2) {
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            // 获取(x, y)位置的像素值
            cv::Vec3b color1 = img1.at<cv::Vec3b>(y, x);
            cv::Vec3b color2 = img2.at<cv::Vec3b>(y, x);
            color2[0] = color1[0] * 0.299;//red
            color2[1] = color1[1] * 0.587;//green
            color2[2] = color1[2] * 0.114;//blue
        }
    }
}
void qiangti() {
    // 读取图像
    //cv::Mat image = cv::imread("房门开-暗.png");
    cv::Mat image = cv::imread("房门开.png",-1);
    // 转换成灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("gray", gray);
    std::cout << gray.channels() << std::endl;
    //cv::Mat gray2;
    //My_BGR2GRAY(image, gray2);
    //cv::imshow("gray2", gray2);

    // 应用阈值操作
    cv::Mat edges;
    auto yuzhi = getgraylightest(gray);  
    std::cout <<(int)yuzhi<<std::endl;
    cv::threshold(gray, edges, yuzhi*1.4, 255, cv::THRESH_BINARY);//cv::THRESH_BINARY | cv::THRESH_TRIANGLE
    cv::Mat edges2= edges.clone();
    cv::imshow("Edges2", edges2);
    // 可选：使用形态学操作加强边界
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(edges, edges, cv:: MORPH_DILATE, kernel, cv::Point(-1, -1), 1);

    // 显示结果
    cv::imshow("Edges", edges);
    cv::waitKey(0);
}

int bluetest() {
    // 加载图片
    cv::Mat image = cv::imread("测试.png");

    // 创建一个与原图相同大小的掩膜，并用0填充
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // 遍历图片中的每个像素
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取(x, y)位置的像素值
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            //std::cout << (int)color[0] << ",";
            // 如果像素是蓝色的，将掩膜中的对应位置设为255
            if ((uchar)color[0] > (uchar)100 && (uchar)color[2] < 120 ) {
                mask.at<uchar>(y, x) = 255;
            }
        }
        //std::cout << std::endl;
    }
    cv::imshow("mask", mask);
    // 使用掩膜提取蓝色的像素
    cv::Mat bluePixels;
    image.copyTo(bluePixels, mask);

    // 显示蓝色的像素
    cv::imshow("Blue Pixels", bluePixels);
    cv::waitKey(0);

    return 0;
}
int bluetest2() {
    // 加载图片
    cv::Mat image = cv::imread("测试2.jpg");

    // 创建一个与原图相同大小的掩膜，并用0填充
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // 遍历图片中的每个像素
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // 获取(x, y)位置的像素值
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            //std::cout << (int)color[0] << ",";
            // 如果像素是蓝色的，将掩膜中的对应位置设为255
            if ((uchar)color[0] > (uchar)100) {
                mask.at<uchar>(y, x) = 255;
            }
        }
        //std::cout << std::endl;
    }
    cv::imshow("mask", mask);
    // 使用掩膜提取蓝色的像素
    cv::Mat bluePixels;
    image.copyTo(bluePixels, mask);

    // 显示蓝色的像素
    cv::imshow("Blue Pixels", bluePixels);
    cv::waitKey(0);

    return 0;
}

void qiangti2() {
    // 读取图像
    cv::Mat image = cv::imread("target.jpg");

    cv::Mat img = image.clone();


    // 转换成灰度图
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 应用阈值操作
    cv::Mat edges;
    cv::threshold(gray, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
    cv::Mat edges2 = edges.clone();
    cv::imshow("Edges2", edges2);

    // 边缘检测
    cv::Mat edges3;
    cv::Canny(edges2, edges3, 0, 250, 3);
    cv::imshow("Edges3", edges3);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges2.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓
    cv::Mat result1 = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        // 用多边形逼近轮廓
        const double epsilon = 0.01; // 0.02表示approxPolyDP的精度
        std::vector<cv::Point> poly;
        cv::approxPolyDP(contours[i], poly, epsilon, true);
        if (poly.size() == 11 && contours[i][0].x > 600) {
            for (size_t j = 0; j < contours[i].size(); j++)
            {
                std::cout << contours[i] << std::endl;
            }
            std::cout <<  std::endl;
            cv::drawContours(result1, contours, static_cast<int>(i), cv::Scalar(255, 255, 255), 1, 8);
        }
 
    }
    // 绘制轮廓
    cv::Mat result2= image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area <= 50)
        {
            cv::drawContours(result2, contours, static_cast<int>(i), cv::Scalar(255, 255, 255), 1, 8);
        }

    }
    // 展示结果
    cv::imshow("Map with Contours1", result1);
    cv::imshow("Map with Contours2", result2);
    cv::waitKey(0);
}
enum roomboxtype
{
    mysite =1,
    gone=2,
    opened=3,
    opened_boss=4,
    boss=5,
    none=6
};
struct ROOMBOX
{
    ROOMBOX() {
        memset(this, 0, sizeof(ROOMBOX));
    }
    ROOMBOX(int px,int py,int width,int hight,int x, int y, roomboxtype type = roomboxtype::gone) {
        this->pt1 = { px,py };
        this->pt2 = { px+ width,py + hight };
        this->x = x;
        this->y = y;
        this->type = type;
    }
    cv::Point pt1;
    cv::Point pt2;
    int x;
    int y;
    roomboxtype type;
};
int tiaozhengliangdu() {
    // 加载图片
    cv::Mat img1 = cv::imread("房门开.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("房门开-暗.png", cv::IMREAD_GRAYSCALE);

    // 检查图片是否成功加载
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Unable to load images." << std::endl;
        return -1;
    }

    // 确保两张图片尺寸相同
    if (img1.size() != img2.size()) {
        std::cerr << "Error: Images must be of the same size." << std::endl;
        return -1;
    }

    // 计算两张图片的直方图均衡化
    cv::Mat histogramEqualizedImg1, histogramEqualizedImg2;
    cv::equalizeHist(img1, histogramEqualizedImg1);
    cv::equalizeHist(img2, histogramEqualizedImg2);

    // 显示结果
    cv::imshow("Original Image 1", img1);
    cv::imshow("Equalized Image 1", histogramEqualizedImg1);
    cv::imshow("Original Image 2", img2);
    cv::imshow("Equalized Image 2", histogramEqualizedImg2);

    // 等待按键事件
    cv::waitKey(0);

    return 0;
}
void adjustBrightness(cv::Mat& image1, cv::Mat& image2, float alpha) {
    // 计算两张图片的亮度平均值
    double mean1 = cv::mean(image1)[0];
    double mean2 = cv::mean(image2)[0];
    float delta = alpha * (mean1 - mean2);

    // 调整亮度
    image1 -= delta;
    image2 += delta;

    // 限制亮度在[0, 255]
    image1 = cv::max(cv::min(image1, 255.0), 0.0);
    image2 = cv::max(cv::min(image2, 255.0), 0.0);
}

double compareHistograms(const cv::Mat& image1, const cv::Mat& image2) {
    // 确保两个图像具有相同的大小和类型
    if (image1.size() != image2.size() || image1.type() != image2.type()) {
        std::cerr << "Error: Images have different sizes or types!" << std::endl;
        return -1.0;
    }

    // 计算两个图像的直方图
    int histSize = 256; // 每个维度的大小
    float range[] = { 0, 256 }; // 灰度级的范围
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat hist1, hist2;
    cv::calcHist(&image1, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange, uniform, accumulate);
    cv::calcHist(&image2, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange, uniform, accumulate);

    // 归一化直方图
    cv::normalize(hist1, hist1, 1, 0, cv::NORM_L1);
    cv::normalize(hist2, hist2, 1, 0, cv::NORM_L1);

    // 使用斯皮尔曼相关系数来比较直方图
    return cv::compareHist(hist1, hist2, cv::HistCompMethods::HISTCMP_KL_DIV);
}

float lightHSL(cv::Mat img)
{
    cv::Scalar scalarHSL = cv::mean(img);
    double imgMax = cv::max(scalarHSL.val[0], scalarHSL.val[1]);
    imgMax = cv::max(imgMax, scalarHSL.val[2]);
    double imgMin = cv::min(scalarHSL.val[0], scalarHSL.val[1]);
    imgMin = cv::min(imgMin, scalarHSL.val[2]);
    float hslLight = (imgMax + imgMin) / 2;
    return hslLight;
}
float lightGray(cv::Mat img)
{
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    cv::Scalar grayScalar = cv::mean(imgGray);
    float imgGrayLight = grayScalar.val[0];
    return imgGrayLight;
}

extern int image_thresholding();
bool IsMysite(cv::Mat& miniroom);
bool IsDoorOpen(cv::Mat& miniroom);
bool IsDoorOpen2(cv::Mat& miniroom); 
bool IsRood(cv::Mat& miniroom);
int main()
{
    //qiangti();
    //return 0;
     //加载目标图像和模板图像
    cv::Mat targetImage = cv::imread("target.jpg");
    cv::Mat templateImage = cv::imread("boss.PNG");

    if (targetImage.empty() || templateImage.empty()) {
        std::cout << "图像加载失败" << std::endl;
        return -1;
    }
    auto t = PictureMatch().getmatchbox(targetImage, templateImage);
    //for (const auto& loc : t)
    //{
    //    std::cout << "传统算法相似度：" << loc.score <<" 对比相似度：" << loc.pixsc << std::endl;
    //    // 标记匹配位置
    //    cv::rectangle(targetImage, cv::Point(loc.x1, loc.y1), cv::Point(loc.x2, loc.y2), cv::Scalar(0, 0, 255));
    //}
    std::cout << "传统算法相似度：" << t.score << " 对比相似度：" << t.pixsc << std::endl;
    cv::Mat targetImagecopy = targetImage.clone();
    cv::rectangle(targetImagecopy, cv::Point(t.x1, t.y1), cv::Point(t.x2, t.y2), cv::Scalar(0, 0, 255));
    cv::imshow("Target Image Copy", targetImagecopy);

    int x_light =t.x1 % templateImage.cols;
    int x_right = targetImage.cols - (targetImage.cols - t.x2)% templateImage.cols;
    int y_down = targetImage.rows - (targetImage.rows - t.y2) % templateImage.rows;
    int y_up = t.y1 % templateImage.rows + templateImage.rows;
    cv::Mat targetImageclone = targetImage.clone();
    cv::rectangle(targetImageclone, cv::Point(x_light, y_up), cv::Point(x_right, y_down), cv::Scalar(0, 0, 255));//画出轮廓范围
    std::vector<std::vector< ROOMBOX>>mapdata(y_down-y_up/ templateImage.rows, std::vector< ROOMBOX>(x_right - x_light / templateImage.cols, ROOMBOX()));
    int boss_x = 0, boss_y = 0, mine_x = 0, mine_y = 0;
    bool inbossroom = true;
    for (int x = 0,x1 = 0; x1 + x_light < x_right; x++, x1= x * templateImage.cols)
    {
        //mapdata.push_back(std::vector< ROOMBOX>());
        for (int y = 0,y1 = 0;  y1 + y_up < y_down; y++, y1=y* templateImage.rows)
        {
        
            if (t.x1 == x1 + x_light && t.y1 == y1 + y_up)
            {
                boss_x = x;
                boss_y = y;
                mapdata[x][y] = ROOMBOX(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows, x, y, roomboxtype::boss);  
                cv::rectangle(targetImageclone, cv::Point(x1 + x_light, y1 + y_up), cv::Point(x1 + x_light + templateImage.cols, y1 + y_up + templateImage.rows), cv::Scalar(0, 255, 255));//画出轮廓范围
            }
            else
            {
                cv::Mat miniroom = targetImage(cv::Rect(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows));
                std::cout << lightGray(miniroom) << std::endl;
           /*     cv::imshow("1", miniroom);
                cv::waitKey(0)*/;
                if (IsMysite(miniroom))
                {
                    mine_x = x;
                    mine_y = y;
                    inbossroom = !inbossroom;
                    mapdata[x][y] = ROOMBOX(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows, x, y, roomboxtype::mysite);
                    cv::rectangle(targetImageclone, cv::Point(x1 + x_light, y1 + y_up), cv::Point(x1 + x_light + templateImage.cols, y1 + y_up + templateImage.rows), cv::Scalar(0, 255, 0));//画出轮廓范围
                }
                else if (IsDoorOpen(miniroom))
                {
                    mapdata[x][y] = ROOMBOX(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows, x, y, roomboxtype::opened);
                    cv::rectangle(targetImageclone, cv::Point(x1 + x_light, y1 + y_up), cv::Point(x1 + x_light + templateImage.cols, y1 + y_up + templateImage.rows), cv::Scalar(255, 0, 0));//画出轮廓范围
                }     
                else if(IsRood(miniroom))
                {
                    mapdata[x][y] = ROOMBOX(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows, x, y, roomboxtype::gone);  
                    cv::rectangle(targetImageclone, cv::Point(x1 + x_light, y1 + y_up), cv::Point(x1 + x_light + templateImage.cols, y1 + y_up + templateImage.rows), cv::Scalar(0, 0, 255));//画出轮廓范围
                }
                else
                {
                    mapdata[x][y] = ROOMBOX(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows, x, y, roomboxtype::none);
                    //mapdata[x].push_back(ROOMBOX(x1 + x_light, y1 + y_up, templateImage.cols, templateImage.rows, x, y, roomboxtype::none));
                }
            }            
        }
    }

    if ((boss_x == mine_x && abs((int)boss_y - (int)mine_y) == 1 || boss_y == mine_y && abs((int)boss_x - (int)mine_x) == 1) && !inbossroom)//如果在boss房旁边
    {
        std::vector<ROOMBOX> recheckbox;
        for (size_t x = 0; x < mapdata.size(); x++)
        {
            for (size_t y = 0; y < mapdata[x].size(); y++)
            {
                if (x == boss_x && y == boss_y || x == mine_x && y == mine_y)//排除当前位置 与boss 房
                    continue;
                if ((x == mine_x && abs((int)y - (int)mine_y) == 1 || y == mine_y && abs((int)x - (int)mine_x) == 1)&& mapdata[x][y].type ==roomboxtype::gone)
                {
                    recheckbox.push_back(mapdata[x][y]);
                }
            }
        }
        //这里写二次门开的检测代码 取当前房间的亮度
        cv::Mat miniroom = targetImage(cv::Rect(mapdata[mine_x][mine_y].pt1.x, mapdata[mine_x][mine_y].pt1.y, templateImage.cols, templateImage.rows));
        cv::imshow("1", miniroom);
        for (size_t i = 0; i < recheckbox.size(); i++)
        {
            std::cout << recheckbox[i].x<< "," << recheckbox[i].y <<std::endl;
        }
    }
    cv::Mat img = targetImage.clone();
    // 转换成灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("gray", gray);
    // 应用阈值操作
    cv::Mat edges;
    cv::threshold(gray, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imshow("Edges2", edges);

    cv::imshow("Target Image clone", targetImageclone);
    cv::imshow("Target Image", targetImage);
    cv::waitKey(0);
    return 0;
}




bool IsMysite(cv::Mat& miniroom) {
    cv::Mat rminiroom = PictureMatch().dealwithblue(miniroom);//蓝色地图的话可以初始化蒙版 直接识别
    static cv::Mat templateImage = cv::imread("我的位置.PNG");
    cv::Mat rtemplateImage = PictureMatch().dealwithblue(templateImage);
    auto r = PictureMatch().SingleChannelPixelComparison(rminiroom, rtemplateImage, RBGCHANNEL::blue);
    return  r> 0.93;
}

//普通的开门
bool IsDoorOpen(cv::Mat& miniroom) {
    static cv::Mat  gray1 = cv::imread("房门开.png"); 
    auto r = PictureMatch().BGR2GRAYCMP(gray1, miniroom, 1.4);
    return  r >0.95;
}
//邻近boss房的开门
bool IsDoorOpen2(cv::Mat& miniroom) {
  
    auto r =0;

    return  r > 0.96;
}

bool IsRood(cv::Mat& miniroom) {
    return 0;
}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
