#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

struct TwistParams {
    double cx;
    double cy;
    double maxRadius;
    double strength;
};

cv::Point2d twistPoint(double x, double y, const TwistParams& p)
{
    double dx = x - p.cx;
    double dy = y - p.cy;

    double r = std::sqrt(dx * dx + dy * dy);

    if (r > p.maxRadius)
    {
        return cv::Point2d(x, y);
    }

    double factor = 1.0 - r / p.maxRadius;
    double angle = p.strength * factor;

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    double x_d = p.cx + dx * cosA - dy * sinA;
    double y_d = p.cy + dx * sinA + dy * cosA;

    return cv::Point2d(x_d, y_d);
}

cv::Point2d untwistPoint(double x_d, double y_d, const TwistParams& p)
{
    double dx = x_d - p.cx;
    double dy = y_d - p.cy;

    double r = std::sqrt(dx * dx + dy * dy);

    if (r > p.maxRadius)
    {
        return cv::Point2d(x_d, y_d);
    }

    double factor = 1.0 - r / p.maxRadius;
    double angle = -p.strength * factor;

    double cosA = std::cos(angle);
    double sinA = std::sin(angle);

    double x = p.cx + dx * cosA - dy * sinA;
    double y = p.cy + dx * sinA + dy * cosA;

    return cv::Point2d(x, y);
}

void buildTwistDistortMap(const cv::Size& size, const TwistParams& p, cv::Mat& mapX, cv::Mat& mapY)
{
    mapX = cv::Mat(size, CV_32FC1);
    mapY = cv::Mat(size, CV_32FC1);

    for (int v = 0; v < size.height; v++)
    {
        for (int u = 0; u < size.width; u++)
        {
            cv::Point2d source = untwistPoint(u, v, p);

            mapX.at<float>(v, u) = static_cast<float>(source.x);
            mapY.at<float>(v, u) = static_cast<float>(source.y);
        }
    }
}

void buildTwistUndistortMap(const cv::Size& size, const TwistParams& p, cv::Mat& mapX, cv::Mat& mapY)
{
    mapX = cv::Mat(size, CV_32FC1);
    mapY = cv::Mat(size, CV_32FC1);

    for (int v = 0; v < size.height; v++)
    {
        for (int u = 0; u < size.width; u++)
        {
            cv::Point2d source = twistPoint(u, v, p);

            mapX.at<float>(v, u) = static_cast<float>(source.x);
            mapY.at<float>(v, u) = static_cast<float>(source.y);
        }
    }
}

int main()
{
    std::string imagePath = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_5/raven2.jpg";

    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
    {
        std::cout << "Cannot open image" << std::endl;
        return 0;
    }

    TwistParams params;

    params.cx = image.cols / 2.0;
    params.cy = image.rows / 2.0;
    params.maxRadius = std::min(image.cols, image.rows) * 0.75;
    params.strength = 2.2;

    cv::Mat distortMapX;
    cv::Mat distortMapY;
    cv::Mat undistortMapX;
    cv::Mat undistortMapY;

    buildTwistDistortMap(image.size(), params, distortMapX, distortMapY);

    cv::Mat distorted;
    cv::remap(image, distorted, distortMapX, distortMapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    buildTwistUndistortMap(image.size(), params, undistortMapX, undistortMapY);

    cv::Mat restored;
    cv::remap(distorted, restored, undistortMapX, undistortMapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::imwrite("original.png", image);
    cv::imwrite("twist_distorted.png", distorted);
    cv::imwrite("twist_restored.png", restored);

    cv::imshow("Original", image);
    cv::imshow("Twist distorted", distorted);
    cv::imshow("Restored", restored);

    cv::waitKey(0);

    return 0;
}