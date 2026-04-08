#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>


//точки у вигляді 2-д подаються у піксельному форматі!
struct InliersData {
    std::vector<cv::Vec3d> points_3d;
    std::vector<cv::Vec2d> points_2d;
    cv::Affine3d pose;
};

std::vector<cv::Vec2d> points_to_pixels(const std::vector<cv::Vec3d>& points_3d, const cv::Affine3d& pose, const cv::Matx33d& calibration_matrix) {
    cv::Matx33d R = pose.rotation();
    cv::Vec3d t = pose.translation();
    std::vector<cv::Vec3d> points_in_camera;
    for (int i = 0; i < points_3d.size(); i++) {
        cv::Vec3d vec_in_camera_system = R * cv::Vec3d(points_3d[i][0], points_3d[i][1], points_3d[i][2]) + t;
        points_in_camera.push_back(vec_in_camera_system);
    }
    std::vector<cv::Vec3d> pixel_3d_coordinates;
    for (int j = 0; j < points_in_camera.size(); j++) {
        cv::Vec3d vec_in_projective_coordinates;
        vec_in_projective_coordinates[0] = points_in_camera[j][0] / points_in_camera[j][2];
        vec_in_projective_coordinates[1] = points_in_camera[j][1] / points_in_camera[j][2];
        vec_in_projective_coordinates[2] = 1;
        cv::Vec3d pixel_point_coordinate = calibration_matrix * vec_in_projective_coordinates;
        pixel_3d_coordinates.push_back(pixel_point_coordinate);
    }
    std::vector<cv::Vec2d> pixel_2d_coordinates;
    for (int i = 0; i < pixel_3d_coordinates.size(); i++) {
        cv::Vec2d pixel_point;
        pixel_point[0] = pixel_3d_coordinates[i][0];
        pixel_point[1] = pixel_3d_coordinates[i][1];
        pixel_2d_coordinates.push_back(pixel_point);
    }
    return pixel_2d_coordinates;
}

bool pnp(const std::vector<cv::Vec3d>& points_3d, const std::vector<cv::Vec2d>& points_2d, const cv::Matx33d& calibration_matrix, cv::Affine3d& pose){
    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(points_3d, points_2d, calibration_matrix, cv::Mat::zeros(4, 1, CV_64F), rvec, tvec, false,cv::SOLVEPNP_AP3P);
    if (ok) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        pose = cv::Affine3d(R, tvec);
    }
    return ok;
}

bool threshold_checker(const cv::Affine3d& pose, const double& squared_threshold, const cv::Vec3d& point_3d, const cv::Vec2d& point_2d, const cv::Matx33d& calibration_matrix) {
    cv::Matx33d R = pose.rotation();
    cv::Vec3d t = pose.translation();
    cv::Vec3d vec_in_camera_system = R * cv::Vec3d(point_3d[0], point_3d[1], point_3d[2]) + t;
    if (vec_in_camera_system[2] <= 0.0) {
        return false;
    }
    cv::Vec3d vec_in_projective_coordinates;
    vec_in_projective_coordinates[0] = vec_in_camera_system[0] / vec_in_camera_system[2];
    vec_in_projective_coordinates[1] = vec_in_camera_system[1] / vec_in_camera_system[2];
    vec_in_projective_coordinates[2] = 1;
    cv::Vec3d pixel_coordinates = calibration_matrix * vec_in_projective_coordinates;
    double error_x = pixel_coordinates[0] - point_2d[0];
    double error_y = pixel_coordinates[1] - point_2d[1];
    double complete_error = error_x * error_x + error_y * error_y;
    if (complete_error > squared_threshold) {
        return false;
    }
    return true;
}

InliersData RANSAC(const int& iter, const double& squared_threshold, const std::vector<cv::Vec3d>& points_3d, const std::vector<cv::Vec2d>& points_2d, const cv::Matx33d& calibration_matrix) {
    std::vector<cv::Vec3d> inliers_3d;
    std::vector<cv::Vec2d> inliers_2d;
    cv::Affine3d pose;
    for (int i = 0; i < iter; i++) {
        int index_point1 = rand() % points_3d.size();
        label1:
        int index_point2 = rand() % points_3d.size();
        if (index_point1 == index_point2) {
            goto label1;
        }
        label2:
        int index_point3 = rand() % points_3d.size();
        if (index_point3 == index_point1 || index_point3 == index_point2) {
            goto label2;
        }
        label3:
        int index_point4 = rand() % points_3d.size();
        if (index_point4 == index_point1 || index_point4 == index_point2 || index_point4 == index_point3) {
            goto label3;
        }
        std::vector<cv::Vec3d> curr_point_3d;
        std::vector<cv::Vec2d> curr_point_2d;
        curr_point_3d.push_back(points_3d[index_point1]);
        curr_point_3d.push_back(points_3d[index_point2]);
        curr_point_3d.push_back(points_3d[index_point3]);
        curr_point_3d.push_back(points_3d[index_point4]);
        curr_point_2d.push_back(points_2d[index_point1]);
        curr_point_2d.push_back(points_2d[index_point2]);
        curr_point_2d.push_back(points_2d[index_point3]);
        curr_point_2d.push_back(points_2d[index_point4]);
        cv::Affine3d curr_pose;
        bool good_pnp = pnp(curr_point_3d, curr_point_2d, calibration_matrix, curr_pose);
        if (!good_pnp) {
            continue;
        }
        std::vector<cv::Vec3d> curr_inliers_3d;
        std::vector<cv::Vec2d> curr_inliers_2d;
        for (int j = 0; j < points_3d.size(); j++) {
            if (threshold_checker(curr_pose, squared_threshold, points_3d[j], points_2d[j], calibration_matrix) ) {
                curr_inliers_3d.push_back(points_3d[j]);
                curr_inliers_2d.push_back(points_2d[j]);
            }
        }
        if (curr_inliers_3d.size() > inliers_3d.size()) {
            inliers_3d = curr_inliers_3d;
            inliers_2d = curr_inliers_2d;
            pose = curr_pose;
        }
    }
    InliersData inliers_data;
    inliers_data.points_3d = inliers_3d;
    inliers_data.points_2d = inliers_2d;
    inliers_data.pose = pose;
    return inliers_data;
}

bool refine_pose_lm(const std::vector<cv::Vec3d>& points_3d, const std::vector<cv::Vec2d>& points_2d, const cv::Matx33d& calibration_matrix,cv::Affine3d& pose) {
    cv::Matx33d R_init = pose.rotation();
    cv::Vec3d t_init = pose.translation();
    cv::Mat R_init_mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R_init_mat.at<double>(i, j) = R_init(i, j);
        }
    }
    cv::Mat rvec, tvec(3, 1, CV_64F);
    cv::Rodrigues(R_init_mat, rvec);
    tvec.at<double>(0, 0) = t_init[0];
    tvec.at<double>(1, 0) = t_init[1];
    tvec.at<double>(2, 0) = t_init[2];
    bool ok = cv::solvePnP(points_3d,points_2d, calibration_matrix, cv::Mat::zeros(4, 1, CV_64F),rvec,tvec,true,cv::SOLVEPNP_ITERATIVE);
    if (!ok) {
        return false;
    }
    cv::solvePnPRefineLM(points_3d, points_2d, calibration_matrix, cv::Mat::zeros(4, 1, CV_64F), rvec, tvec);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    pose = cv::Affine3d(R, tvec);
    return true;
}
int main() {
    cv::Matx33d calibration_matrix(
        800.0, 0.0, 320.0,
        0.0, 800.0, 240.0,
        0.0, 0.0, 1.0
    );
    cv::Matx33d R_true(
        0.8528685319524433, -0.5, 0.1503837331804353,
        0.492403876506104, 0.8660254037844386, 0.08682408883346517,
        -0.17364817766693033, 0.0, 0.984807753012208
    );
    cv::Vec3d t_true(0.2, -0.1, 5.0);
    cv::Affine3d true_pose(R_true, t_true);

    std::vector<cv::Vec3d> points_3d;
    points_3d.push_back(cv::Vec3d(-1.0, -1.0, 0.0));
    points_3d.push_back(cv::Vec3d(1.0, -1.0, 0.0));
    points_3d.push_back(cv::Vec3d(1.0, 1.0, 0.0));
    points_3d.push_back(cv::Vec3d(-1.0, 1.0, 0.0));
    points_3d.push_back(cv::Vec3d(-0.5, -0.5, 1.0));
    points_3d.push_back(cv::Vec3d(0.5, -0.5, 1.0));
    points_3d.push_back(cv::Vec3d(0.5, 0.5, 1.0));
    points_3d.push_back(cv::Vec3d(-0.5, 0.5, 1.0));
    points_3d.push_back(cv::Vec3d(-0.8, 0.3, 0.5));
    points_3d.push_back(cv::Vec3d(0.7, -0.2, 0.8));

    std::vector<cv::Vec2d> points_2d = points_to_pixels(points_3d, true_pose, calibration_matrix);

    points_2d[2][0] += 120.0;
    points_2d[2][1] -= 90.0;
    points_2d[7][0] -= 150.0;
    points_2d[7][1] += 110.0;
    int iter = 500;
    double squared_threshold = 25.0;
    InliersData inliers_data = RANSAC(iter, squared_threshold, points_3d, points_2d, calibration_matrix);
    std::cout << "inliers count = " << inliers_data.points_3d.size() << std::endl;
    bool ok = refine_pose_lm(inliers_data.points_3d, inliers_data.points_2d, calibration_matrix, inliers_data.pose);

    if (!ok) {
        std::cout << "refine failed" << std::endl;
        return 0;
    }
    cv::Matx33d R_est = inliers_data.pose.rotation();
    cv::Vec3d t_est = inliers_data.pose.translation();
    std::cout << "true translation: "
              << t_true[0] << " "
              << t_true[1] << " "
              << t_true[2] << std::endl;
    std::cout << "estimated translation: "
              << t_est[0] << " "
              << t_est[1] << " "
              << t_est[2] << std::endl;
    std::cout << "true rotation:" << std::endl;
    std::cout << R_true(0, 0) << " " << R_true(0, 1) << " " << R_true(0, 2) << std::endl;
    std::cout << R_true(1, 0) << " " << R_true(1, 1) << " " << R_true(1, 2) << std::endl;
    std::cout << R_true(2, 0) << " " << R_true(2, 1) << " " << R_true(2, 2) << std::endl;
    std::cout << "estimated rotation:" << std::endl;
    std::cout << R_est(0, 0) << " " << R_est(0, 1) << " " << R_est(0, 2) << std::endl;
    std::cout << R_est(1, 0) << " " << R_est(1, 1) << " " << R_est(1, 2) << std::endl;
    std::cout << R_est(2, 0) << " " << R_est(2, 1) << " " << R_est(2, 2) << std::endl;
    return 0;
}