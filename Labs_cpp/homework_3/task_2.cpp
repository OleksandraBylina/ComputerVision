#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>


struct LinearResidual{
    LinearResidual(double x, double y): x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const{
        residual[0] = T(y_) - a[0] * T(x_) - b[0];
        return true;
    }
private:
    double x_;
    double y_;

};

struct Point {
    double x;
    double y;
};

struct Line {
    double A;
    double B;
    double C;
};

Line LinearRegression(const std::vector<Point>& points) {
    ceres::Problem problem;
    double a = 0.0;
    double b = 0.0;
    for (int i = 0; i < points.size(); ++i) {
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<LinearResidual, 1, 1, 1>(new LinearResidual(points[i].x, points[i].y));
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &a, &b);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    Line line;
    line.A = a;
    line.B = -1;
    line.C = b;
    return line;
}
void draw_plot(const std::string& name,
               const std::vector<Point>& points,
               double true_a, double true_b,
               double found_a, double found_b) {
    int width = 900;
    int height = 700;
    int margin = 60;

    double min_x = points[0].x;
    double max_x = points[0].x;
    double min_y = points[0].y;
    double max_y = points[0].y;

    for (int i = 0; i < points.size(); i++) {
        if (points[i].x < min_x) min_x = points[i].x;
        if (points[i].x > max_x) max_x = points[i].x;
        if (points[i].y < min_y) min_y = points[i].y;
        if (points[i].y > max_y) max_y = points[i].y;
    }

    double y1_true = true_a * min_x + true_b;
    double y2_true = true_a * max_x + true_b;
    double y1_found = found_a * min_x + found_b;
    double y2_found = found_a * max_x + found_b;

    min_y = std::min(min_y, std::min(y1_true, y1_found));
    max_y = std::max(max_y, std::max(y2_true, y2_found));

    double dx = max_x - min_x;
    double dy = max_y - min_y;

    min_x -= 0.1 * dx;
    max_x += 0.1 * dx;
    min_y -= 0.1 * dy;
    max_y += 0.1 * dy;

    cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    auto to_img = [&](double x, double y) {
        int px = margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin);
        int py = height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin);
        return cv::Point(px, py);
    };

    for (int i = 0; i < points.size(); i++) {
        cv::circle(img, to_img(points[i].x, points[i].y), 4, cv::Scalar(0, 0, 0), -1);
    }

    cv::line(img,
             to_img(min_x, true_a * min_x + true_b),
             to_img(max_x, true_a * max_x + true_b),
             cv::Scalar(0, 180, 0), 2);

    cv::line(img,
             to_img(min_x, found_a * min_x + found_b),
             to_img(max_x, found_a * max_x + found_b),
             cv::Scalar(0, 0, 255), 2);

    cv::putText(img, "green - true line", cv::Point(30, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 180, 0), 2);
    cv::putText(img, "red - found line", cv::Point(30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

    cv::imshow(name, img);
    cv::imwrite(name + ".png", img);
}


int main() {
    std::vector<Point> points;

    double true_a = 2.0;
    double true_b = 1.0;

    for (int i = -10; i <= 10; i++) {
        Point p;
        p.x = i;
        p.y = true_a * i + true_b;

        if (i % 3 == 0) {
            p.y += 0.3;
        }
        if (i % 4 == 0) {
            p.y -= 0.2;
        }

        points.push_back(p);
    }

    Point outlier1;
    outlier1.x = -8.0;
    outlier1.y = 20.0;
    points.push_back(outlier1);

    Point outlier2;
    outlier2.x = 0.0;
    outlier2.y = -15.0;
    points.push_back(outlier2);

    Point outlier3;
    outlier3.x = 7.0;
    outlier3.y = 30.0;
    points.push_back(outlier3);

    Point outlier4;
    outlier4.x = -3.0;
    outlier4.y = 16.0;
    points.push_back(outlier4);

    Line line = LinearRegression(points);

    double found_a = -line.A / line.B;
    double found_b = -line.C / line.B;

    std::cout << "true line:  y = " << true_a << " * x + " << true_b << std::endl;
    std::cout << "found line: y = " << found_a << " * x + " << found_b << std::endl;
    std::cout << "general form: "
              << line.A << " * x + "
              << line.B << " * y + "
              << line.C << " = 0" << std::endl;

    draw_plot("linear_regression_test", points, true_a, true_b, found_a, found_b);

    cv::waitKey(0);
    return 0;
}