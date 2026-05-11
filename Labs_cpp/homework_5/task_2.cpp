#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>

struct State {
    double x;
    double y;
    double vx;
    double vy;
    double ax;
    double ay;
};

State trueState(double t)
{
    State s;

    s.x = 2.0 * t + 0.5 * std::sin(0.5 * t);
    s.y = 1.0 * t + 2.0 * std::cos(0.3 * t);

    s.vx = 2.0 + 0.25 * std::cos(0.5 * t);
    s.vy = 1.0 - 0.6 * std::sin(0.3 * t);

    s.ax = -0.125 * std::sin(0.5 * t);
    s.ay = -0.18 * std::cos(0.3 * t);

    return s;
}

double addNoise(double value, double sigma, std::default_random_engine& generator)
{
    std::normal_distribution<double> distribution(0.0, sigma);
    return value + distribution(generator);
}

cv::Point toImagePoint(double x, double y, double scale, int height)
{
    int u = static_cast<int>(x * scale + 50.0);
    int v = static_cast<int>(height - (y * scale + 80.0));

    return cv::Point(u, v);
}

int main()
{
    double dt = 0.01;
    double totalTime = 20.0;

    int steps = static_cast<int>(totalTime / dt);
    int gpsPeriod = 20;

    double accelNoiseSigma = 0.03;
    double gpsNoiseSigma = 0.4;

    std::default_random_engine generator(10);

    cv::KalmanFilter kf(4, 2, 2, CV_64F);

    kf.transitionMatrix = (cv::Mat_<double>(4, 4) <<
        1.0, 0.0, dt, 0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);

    kf.controlMatrix = (cv::Mat_<double>(4, 2) <<
        0.5 * dt * dt, 0.0,
        0.0, 0.5 * dt * dt,
        dt, 0.0,
        0.0, dt);

    kf.measurementMatrix = (cv::Mat_<double>(2, 4) <<
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0);

    kf.processNoiseCov = (cv::Mat_<double>(4, 4) <<
        1e-5, 0.0, 0.0, 0.0,
        0.0, 1e-5, 0.0, 0.0,
        0.0, 0.0, 1e-4, 0.0,
        0.0, 0.0, 0.0, 1e-4);

    kf.measurementNoiseCov = (cv::Mat_<double>(2, 2) <<
        1.5, 0.0,
        0.0, 1.5);

    cv::setIdentity(kf.errorCovPost, cv::Scalar(0.1));

    State initial = trueState(0.0);

    kf.statePost = (cv::Mat_<double>(4, 1) <<
        initial.x,
        initial.y,
        initial.vx,
        initial.vy);

    std::vector<cv::Point2d> trueTrajectory;
    std::vector<cv::Point2d> estimatedTrajectory;
    std::vector<cv::Point2d> gpsTrajectory;

    double sumError = 0.0;
    double maxError = 0.0;

    for (int i = 0; i <= steps; i++)
    {
        double t = i * dt;

        State real = trueState(t);

        double measuredAx = addNoise(real.ax, accelNoiseSigma, generator);
        double measuredAy = addNoise(real.ay, accelNoiseSigma, generator);

        cv::Mat control = (cv::Mat_<double>(2, 1) << measuredAx, measuredAy);

        kf.predict(control);

        if (i % gpsPeriod == 0)
        {
            double gpsX = addNoise(real.x, gpsNoiseSigma, generator);
            double gpsY = addNoise(real.y, gpsNoiseSigma, generator);

            cv::Mat measurement = (cv::Mat_<double>(2, 1) << gpsX, gpsY);

            kf.correct(measurement);

            gpsTrajectory.push_back(cv::Point2d(gpsX, gpsY));
        }

        double estimatedX = kf.statePost.at<double>(0);
        double estimatedY = kf.statePost.at<double>(1);

        double error = std::sqrt(
            (estimatedX - real.x) * (estimatedX - real.x) +
            (estimatedY - real.y) * (estimatedY - real.y));

        sumError += error;

        if (error > maxError)
        {
            maxError = error;
        }

        trueTrajectory.push_back(cv::Point2d(real.x, real.y));
        estimatedTrajectory.push_back(cv::Point2d(estimatedX, estimatedY));
    }

    double meanError = sumError / static_cast<double>(steps + 1);

    std::cout << "Mean error: " << meanError << std::endl;
    std::cout << "Max error: " << maxError << std::endl;

    int width = 1000;
    int height = 700;
    double scale = 35.0;

    cv::Mat result(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 1; i < trueTrajectory.size(); i++)
    {
        cv::Point p1 = toImagePoint(trueTrajectory[i - 1].x, trueTrajectory[i - 1].y, scale, height);
        cv::Point p2 = toImagePoint(trueTrajectory[i].x, trueTrajectory[i].y, scale, height);

        cv::line(result, p1, p2, cv::Scalar(0, 180, 0), 2);
    }

    for (int i = 1; i < estimatedTrajectory.size(); i++)
    {
        cv::Point p1 = toImagePoint(estimatedTrajectory[i - 1].x, estimatedTrajectory[i - 1].y, scale, height);
        cv::Point p2 = toImagePoint(estimatedTrajectory[i].x, estimatedTrajectory[i].y, scale, height);

        cv::line(result, p1, p2, cv::Scalar(255, 0, 0), 2);
    }

    for (int i = 0; i < gpsTrajectory.size(); i++)
    {
        cv::Point p = toImagePoint(gpsTrajectory[i].x, gpsTrajectory[i].y, scale, height);

        cv::circle(result, p, 3, cv::Scalar(0, 0, 255), -1);
    }

    cv::putText(result, "Green: true trajectory", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 180, 0), 2);
    cv::putText(result, "Blue: Kalman estimation", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    cv::putText(result, "Red: GPS measurements", cv::Point(30, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

    cv::imwrite("kalman_result.png", result);

    cv::imshow("Kalman Filter Result", result);
    cv::waitKey(0);

    return 0;
}