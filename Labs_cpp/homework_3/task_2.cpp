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



int main() {
    std::vector<Point> points;

    for (int i = -10; i <= 10; i++) {
        Point p;
        p.x = i;
        p.y = 2.0 * i + 1.0;

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

    Line line = LinearRegression(points);

    std::cout << "line: " << line.A << " * x + " << line.C << std::endl;
    std::cout << "general form: "
              << line.A << " * x + "
              << line.B << " * y + "
              << line.C << " = 0" << std::endl;

    return 0;
}















