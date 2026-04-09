#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

struct Point {
    double x;
    double y;
};

struct Line {
    double A;
    double B;
    double C;
};

double s_xy(const std::vector<Point>& inlier_points) {
    double sum = 0;
    for (int i = 0; i < inlier_points.size(); i++) {
        sum += inlier_points[i].x * inlier_points[i].y;
    }
    return sum;
}

double s_x(const std::vector<Point>& inlier_points) {
    double sum = 0;
    for (int i = 0; i < inlier_points.size(); i++) {
        sum += inlier_points[i].x;
    }
    return sum;
}

double s_y(const std::vector<Point>& inlier_points) {
    double sum = 0;
    for (int i = 0; i < inlier_points.size(); i++) {
        sum += inlier_points[i].y;
    }
    return sum;
}

double s_xx(const std::vector<Point>& inlier_points) {
    double sum = 0;
    for (int i = 0; i < inlier_points.size(); i++) {
        sum += inlier_points[i].x * inlier_points[i].x;
    }
    return sum;
}

Line line_from_two_points(Point p1, Point p2) {
    Line line;
    line.A = p1.y - p2.y;
    line.B = p2.x - p1.x;
    line.C = p1.x * p2.y - p2.x * p1.y;
    return line;
}


std::vector<Point> RANSAC(const int& iter, const double& threshold, const std::vector<Point>& all_points) {
    std::vector<std::vector<Point>> point_pairs;
    for (int i = 0; i < all_points.size(); i++) {
        for (int j = i + 1; j < all_points.size(); j++) {
            std::vector<Point> one_pair;
            one_pair.push_back(all_points[i]);
            one_pair.push_back(all_points[j]);
            point_pairs.push_back(one_pair);
        }
    }
    std::vector<Point> inliers_for_best_pair;
    for (int i = 0; i < iter; i++) {
        int index_pair = rand() % point_pairs.size();
        Line line = line_from_two_points(point_pairs[index_pair][0], point_pairs[index_pair][1]);
        point_pairs.erase(point_pairs.begin() + index_pair);
        std::vector<Point> local_inliers;
        for (int j = 0; j < all_points.size(); j++) {
            if (std::abs(line.B) < 1e-12) {
                continue;
            }
            double curr_a = -line.A / line.B;
            double curr_b = -line.C / line.B;
            double curr_threshold = std::abs(all_points[j].y - (curr_a * all_points[j].x + curr_b));
            if (curr_threshold < threshold) {
                local_inliers.push_back(all_points[j]);
            }
        }
        if (local_inliers.size() > inliers_for_best_pair.size()) {
            inliers_for_best_pair = local_inliers;
        }
    }
    return inliers_for_best_pair;

}

Line linear_regression(const std::vector<Point>& all_points, const int& iter, const double& threshold) {
    label:
    std::vector<Point> inlier_points = RANSAC(iter, threshold, all_points);
    int n = inlier_points.size();
    double ss_x = s_x(inlier_points);
    double ss_y = s_y(inlier_points);
    double ss_xx = s_xx(inlier_points);
    double ss_xy = s_xy(inlier_points);
    double ss_x2 = ss_x * ss_x;
    double denom = n * ss_xx - ss_x2;
    if (denom - 0.00001 < 0) {
        goto label;
        // return linear_regression(all_points, iter, threshold);
    }
    double a = (n * ss_xy - ss_x * ss_y) / denom;
    double b = ((ss_xx * ss_y - ss_x * ss_xy)) / denom;
    Line line;
    line.A = a;
    line.B = -1;
    line.C = b;
    return line;
}

void draw_plot(const std::string& name,
               const std::vector<Point>& points,
               double true_a, double true_b,
               double found_a, double found_b,
               double found_a_without_ransac, double found_b_without_ransac) {
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
    double y1_found_without_ransac = found_a_without_ransac * min_x + found_b_without_ransac;
    double y2_found_without_ransac = found_a_without_ransac * max_x + found_b_without_ransac;

    min_y = std::min(min_y, std::min(y1_true, std::min(y1_found, y1_found_without_ransac)));
    max_y = std::max(max_y, std::max(y2_true, std::max(y2_found, y2_found_without_ransac)));

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
             to_img(min_x, found_a_without_ransac * min_x + found_b_without_ransac),
             to_img(max_x, found_a_without_ransac * max_x + found_b_without_ransac),
             cv::Scalar(255, 0, 0), 2);

    cv::line(img,
             to_img(min_x, found_a * min_x + found_b),
             to_img(max_x, found_a * max_x + found_b),
             cv::Scalar(0, 0, 255), 2);

    cv::putText(img, "green - true line", cv::Point(30, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 180, 0), 2);
    cv::putText(img, "blue - without RANSAC", cv::Point(30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    cv::putText(img, "red - with RANSAC", cv::Point(30, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

    cv::imshow(name, img);
    cv::imwrite(name + ".png", img);
}

Line linear_regression_without_ransac(const std::vector<Point>& all_points) {
    int n = all_points.size();
    double ss_x = s_x(all_points);
    double ss_y = s_y(all_points);
    double ss_xx = s_xx(all_points);
    double ss_xy = s_xy(all_points);
    double ss_x2 = ss_x * ss_x;
    double denom = n * ss_xx - ss_x2;
    double a = (n * ss_xy - ss_x * ss_y) / denom;
    double b = ((ss_xx * ss_y - ss_x * ss_xy)) / denom;
    Line line;
    line.A = a;
    line.B = -1;
    line.C = b;
    return line;
}

int main() {
    std::srand(42);
    std::mt19937 gen(42);

    struct TestCase {
        std::string name;
        double true_a;
        double true_b;
        int n_inliers;
        int n_outliers;
        double x_min;
        double x_max;
        double noise_sigma;
        double outlier_y_min;
        double outlier_y_max;
        int iter;
        double threshold;
    };

    auto make_dataset = [&](const TestCase& test) {
        std::vector<Point> points;
        std::uniform_real_distribution<double> dist_x(test.x_min, test.x_max);
        std::normal_distribution<double> noise(0.0, test.noise_sigma);
        std::uniform_real_distribution<double> dist_out_y(test.outlier_y_min, test.outlier_y_max);

        for (int i = 0; i < test.n_inliers; i++) {
            double x = dist_x(gen);
            double y = test.true_a * x + test.true_b + noise(gen);
            points.push_back({x, y});
        }

        while (points.size() < test.n_inliers + test.n_outliers) {
            double x = dist_x(gen);
            double y = dist_out_y(gen);
            double dist = std::abs(y - (test.true_a * x + test.true_b));
            if (dist > 3.0 * test.noise_sigma) {
                points.push_back({x, y});
            }
        }

        return points;
    };

    TestCase test = {"test_1", 1.5, 2.0, 35, 10, -10, 10, 0.25, -18, 18, 40, 0.7};

    std::vector<Point> points = make_dataset(test);
    Line result = linear_regression(points, test.iter, test.threshold);
    Line result_without_ransac = linear_regression_without_ransac(points);

    double found_a = -result.A / result.B;
    double found_b = -result.C / result.B;
    double found_a_without_ransac = -result_without_ransac.A / result_without_ransac.B;
    double found_b_without_ransac = -result_without_ransac.C / result_without_ransac.B;

    draw_plot(test.name, points, test.true_a, test.true_b, found_a, found_b, found_a_without_ransac, found_b_without_ransac);

    std::cout << test.name << '\n';
    std::cout << "true line:             y = " << test.true_a << " * x + " << test.true_b << '\n';
    std::cout << "found line with RANSAC:y = " << found_a << " * x + " << found_b << '\n';
    std::cout << "found line without:    y = " << found_a_without_ransac << " * x + " << found_b_without_ransac << '\n';
    std::cout << "total points: " << points.size() << "\n\n";

    cv::waitKey(0);

    return 0;
}