//
// Created by Олександра Биліна on 15.03.2026.
//

#include "basic_operations.h"

#include <opencv2/opencv.hpp>
#include <string>

void basicImageOperations()
{
    std::string path = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/raven.jpg";
    cv::Mat image = cv::imread(path);

    if (image.empty()) {
        std::cout << "Image not loaded!\n";
        return;
    }
    cv::imshow("Original picture", image);
    cv::waitKey(0);
    cv::Mat grey_image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    cv::imshow("Grey picture", grey_image);
    cv::waitKey(0);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(320, 320));
    cv::imshow("Resized picture", resized_image);
    cv::waitKey(0);
    cv::Mat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(9, 9), 0);
    cv::imshow("Blurred picture", blurred_image);
    cv::waitKey(0);
    cv::Mat edges;
    cv::Canny(grey_image, edges, 80, 150);
    cv::imshow("Edges", edges);
    cv::waitKey(0);
    cv::Mat rotated_image;
    cv::rotate(image, rotated_image, cv::ROTATE_90_CLOCKWISE);
    cv::imshow("Rotated picture", rotated_image);
    cv::waitKey(0);
    cv::Mat cropped_image = image(cv::Rect(100, 100, 250, 250));
    cv::imshow("Cropped picture", cropped_image);
    cv::waitKey(0);
    cv::Mat brighter_image = image.clone();
    for (int i = 0; i < brighter_image.rows; ++i)
        for (int j = 0; j < brighter_image.cols; ++j)
            for (int c = 0; c < 3; ++c)
                brighter_image.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(brighter_image.at<cv::Vec3b>(i, j)[c] + 50);
    cv::imshow("Brighter picture", brighter_image);
    cv::waitKey(0);
}

void createImageEx()
{
    cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);
    for (size_t i = 100; i < 200; ++i)
        for (size_t j = 100; j < 200; ++j)
            image.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
    cv::imshow("Square", image);
    cv::waitKey(0);
}


int main()
{
    basicImageOperations();
    createImageEx();
    return 0;
}