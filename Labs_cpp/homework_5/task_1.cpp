#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

struct FisheyeParams {
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double k4;
};

double distortTheta(double theta, const FisheyeParams& p)
{
    double theta2 = theta * theta;
    double theta4 = theta2 * theta2;
    double theta6 = theta4 * theta2;
    double theta8 = theta4 * theta4;

    return theta * (1.0 + p.k1 * theta2 + p.k2 * theta4 + p.k3 * theta6 + p.k4 * theta8);
}

double undistortTheta(double theta_d, const FisheyeParams& p)
{
    double theta = theta_d;

    for (int i = 0; i < 10; i++)
    {
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta4 * theta2;
        double theta8 = theta4 * theta4;

        double f = theta * (1.0 + p.k1 * theta2 + p.k2 * theta4 + p.k3 * theta6 + p.k4 * theta8) - theta_d;
        double df = 1.0 + 3.0 * p.k1 * theta2 + 5.0 * p.k2 * theta4 + 7.0 * p.k3 * theta6 + 9.0 * p.k4 * theta8;

        theta = theta - f / df;
    }

    return theta;
}

void buildDistortMap(const cv::Size& size, const FisheyeParams& p, cv::Mat& mapX, cv::Mat& mapY)
{
    mapX = cv::Mat(size, CV_32FC1);
    mapY = cv::Mat(size, CV_32FC1);

    for (int v = 0; v < size.height; v++)
    {
        for (int u = 0; u < size.width; u++)
        {
            double x_d = (u - p.cx) / p.fx;
            double y_d = (v - p.cy) / p.fy;

            double r_d = std::sqrt(x_d * x_d + y_d * y_d);

            double x = x_d;
            double y = y_d;

            if (r_d > 1e-8)
            {
                double theta_d = r_d;
                double theta = undistortTheta(theta_d, p);
                double r = std::tan(theta);
                double scale = r / r_d;

                x = x_d * scale;
                y = y_d * scale;
            }

            double srcU = p.fx * x + p.cx;
            double srcV = p.fy * y + p.cy;

            mapX.at<float>(v, u) = static_cast<float>(srcU);
            mapY.at<float>(v, u) = static_cast<float>(srcV);
        }
    }
}

void buildUndistortMap(const cv::Size& size, const FisheyeParams& p, cv::Mat& mapX, cv::Mat& mapY)
{
    mapX = cv::Mat(size, CV_32FC1);
    mapY = cv::Mat(size, CV_32FC1);

    for (int v = 0; v < size.height; v++)
    {
        for (int u = 0; u < size.width; u++)
        {
            double x = (u - p.cx) / p.fx;
            double y = (v - p.cy) / p.fy;

            double r = std::sqrt(x * x + y * y);

            double x_d = x;
            double y_d = y;

            if (r > 1e-8)
            {
                double theta = std::atan(r);
                double theta_d = distortTheta(theta, p);
                double scale = theta_d / r;

                x_d = x * scale;
                y_d = y * scale;
            }

            double srcU = p.fx * x_d + p.cx;
            double srcV = p.fy * y_d + p.cy;

            mapX.at<float>(v, u) = static_cast<float>(srcU);
            mapY.at<float>(v, u) = static_cast<float>(srcV);
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

    FisheyeParams params;

    params.fx = image.cols * 0.8;
    params.fy = image.cols * 0.8;
    params.cx = image.cols / 2.0;
    params.cy = image.rows / 2.0;

    params.k1 = 0.25;
    params.k2 = 0.05;
    params.k3 = 0.0;
    params.k4 = 0.0;

    cv::Mat distortMapX;
    cv::Mat distortMapY;
    cv::Mat undistortMapX;
    cv::Mat undistortMapY;

    buildDistortMap(image.size(), params, distortMapX, distortMapY);

    cv::Mat distorted;
    cv::remap(image, distorted, distortMapX, distortMapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    buildUndistortMap(image.size(), params, undistortMapX, undistortMapY);

    cv::Mat restored;
    cv::remap(distorted, restored, undistortMapX, undistortMapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::imwrite("original.png", image);
    cv::imwrite("distorted.png", distorted);
    cv::imwrite("restored.png", restored);

    cv::imshow("Original", image);
    cv::imshow("Distorted", distorted);
    cv::imshow("Restored", restored);

    cv::waitKey(0);

    return 0;
}