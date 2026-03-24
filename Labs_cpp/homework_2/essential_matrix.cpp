
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::vector<cv::Vec3d> make_camera_coords(const std::vector<cv::Vec3d>& coords, const cv::Affine3d& pose) {
    cv::Matx33d R = pose.rotation();
    cv::Vec3d t = pose.translation();
    std::vector<cv::Vec3d> new_coords;
    for (int i = 0; i < coords.size(); i++) {
        cv::Vec3d new_camera_coords = (R * coords[i]) + t;
        new_coords.push_back(new_camera_coords);
    }
    return new_coords;
}


std::vector<cv::Vec2d> make_3d_to_2d (const std::vector<cv::Vec3d>& coords) {
    std::vector<cv::Vec2d> flat_coords;
    for (int i = 0 ; i < coords.size() ; i ++) {
        double x = coords[i][0] / coords[i][2];
        double y = coords[i][1] / coords[i][2];
        cv::Vec2d point = cv::Vec2d(x, y);
        flat_coords.push_back(point);
    }
    return flat_coords;
}

std::vector<cv::Vec3d> make_3_coord (const std::vector<cv::Vec2d>& coords) {
    std::vector<cv::Vec3d> new_coords;
    for (int i = 0 ; i < coords.size() ; i ++) {
        cv::Vec3d point = cv::Vec3d(coords[i][0], coords[i][1], 1);
        new_coords.push_back(point);
    }
    return new_coords;
}

std::vector<cv::Vec2d> make_2_coord (const std::vector<cv::Vec3d>& coords) {
    std::vector<cv::Vec2d> new_coords;
    for (int i = 0 ; i < coords.size() ; i ++) {
        cv::Vec2d point = cv::Vec2d(coords[i][0], coords[i][1]);
        new_coords.push_back(point);
    }
    return new_coords;
}


std::vector<cv::Vec3d> make_pixels (const std::vector<cv::Vec3d>& coords, const cv::Matx33d& calibration_matrix) {
    std::vector<cv::Vec3d> pixel_coords;
    for (int i = 0 ; i < coords.size() ; i ++) {
        cv::Vec3d point = calibration_matrix * coords[i];
        pixel_coords.push_back(point);
    }
    return pixel_coords;
}

std::vector<cv::Vec3d> make_flat_reverse(const std::vector<cv::Vec3d>& coords, const cv::Matx33d& calibration_matrix) {
    cv::Matx33d inv_calibration_matrix = calibration_matrix.inv();
    std::vector<cv::Vec3d> flat_coords;
    for (int i = 0 ; i < coords.size() ; i ++) {
        cv::Vec3d point = inv_calibration_matrix * coords[i];
        flat_coords.push_back(point);
    }
    return flat_coords;
}

std::vector<std::vector<cv::Vec2d>> make_data(const std::vector<cv::Vec3d>& coords, const cv::Affine3d& pose) {
    std::vector<cv::Vec3d> camera2_coords = make_camera_coords(coords, pose);
    std::vector<cv::Vec2d> camera1_flat_coords = make_3d_to_2d(coords);
    std::vector<cv::Vec2d> camera2_flat_coords = make_3d_to_2d(camera2_coords);
    std::vector<std::vector<cv::Vec2d>> full_data = {camera1_flat_coords, camera2_flat_coords};
    return full_data;

}


cv::Mat find_essential_matrix(const std::vector<cv::Vec3d>& coords, const cv::Matx33d& calibration_matrix, const cv::Affine3d& pose) {
    std::vector<std::vector<cv::Vec2d>> norm_coords = make_data(coords, pose);
    std::vector<cv::Vec2d> point_set_camera1_flat = norm_coords[0];
    std::vector<cv::Vec2d> point_set_camera2_flat = norm_coords[1];
    std::vector<cv::Vec3d> point_set_camera1_3 = make_3_coord(point_set_camera1_flat);
    std::vector<cv::Vec3d> point_set_camera2_3 = make_3_coord(point_set_camera2_flat);
    std::vector<cv::Vec3d> point_set_camera1_pixels_3 = make_pixels(point_set_camera1_3, calibration_matrix);
    std::vector<cv::Vec3d> point_set_camera2_pixels_3 = make_pixels(point_set_camera2_3, calibration_matrix);
    std::vector<cv::Vec2d> point_set_camera1 = make_2_coord(point_set_camera1_pixels_3);
    std::vector<cv::Vec2d> point_set_camera2 = make_2_coord(point_set_camera2_pixels_3);
    std::vector<cv::Point2d> points_camera1;
    std::vector<cv::Point2d> points_camera2;
    for (int i = 0; i < point_set_camera1.size(); i++) {
        cv::Vec2d vec_point_camera1 = point_set_camera1[i];
        cv::Point2d point_camera1(vec_point_camera1[0], vec_point_camera1[1]);
        points_camera1.push_back(point_camera1);
        cv::Vec2d vec_point_camera2 = point_set_camera2[i];
        cv::Point2d point_camera2(vec_point_camera2[0], vec_point_camera2[1]);
        points_camera2.push_back(point_camera2);
    }
    cv::Mat essential_matrix = cv::findEssentialMat(points_camera1, points_camera2, cv::Mat(calibration_matrix));
    return essential_matrix;
}

cv::Mat find_fundamental_matrix(const std::vector<cv::Vec3d>& coords, const cv::Matx33d& calibration_matrix, const cv::Affine3d& pose) {
    cv::Mat essential_matrix = find_essential_matrix(coords, calibration_matrix, pose);
    cv::Matx33d inv_matrix = calibration_matrix.inv();
    cv::Matx33d inv_trans_matrix = inv_matrix.t();
    cv::Mat fundamental_matrix = inv_trans_matrix * essential_matrix * inv_matrix;
    return fundamental_matrix;
}

cv::Affine3d find_pose(const std::vector<cv::Vec3d>& coords, const cv::Matx33d& calibration_matrix, const cv::Affine3d& pose) {
    cv::Mat essential_matrix = find_essential_matrix(coords, calibration_matrix, pose);
    std::vector<std::vector<cv::Vec2d>> flat_coords = make_data(coords, pose);
    std::vector<cv::Vec2d> point_set_camera1_flat = flat_coords[0];
    std::vector<cv::Vec2d> point_set_camera2_flat = flat_coords[1];
    std::vector<cv::Vec3d> point_set_camera1_3 = make_3_coord(point_set_camera1_flat);
    std::vector<cv::Vec3d> point_set_camera2_3 = make_3_coord(point_set_camera2_flat);
    std::vector<cv::Vec3d> point_set_camera1_pixels_3 = make_pixels(point_set_camera1_3, calibration_matrix);
    std::vector<cv::Vec3d> point_set_camera2_pixels_3 = make_pixels(point_set_camera2_3, calibration_matrix);
    std::vector<cv::Vec2d> point_set_camera1 = make_2_coord(point_set_camera1_pixels_3);
    std::vector<cv::Vec2d> point_set_camera2 = make_2_coord(point_set_camera2_pixels_3);
    std::vector<cv::Point2d> points_camera1;
    std::vector<cv::Point2d> points_camera2;
    for (int i = 0; i < point_set_camera1.size(); i++) {
        cv::Vec2d vec_point_camera1 = point_set_camera1[i];
        cv::Point2d point_camera1(vec_point_camera1[0], vec_point_camera1[1]);
        points_camera1.push_back(point_camera1);

        cv::Vec2d vec_point_camera2 = point_set_camera2[i];
        cv::Point2d point_camera2(vec_point_camera2[0], vec_point_camera2[1]);
        points_camera2.push_back(point_camera2);
    }

    cv::Mat R, t;
    cv::recoverPose(essential_matrix, points_camera1, points_camera2, cv::Mat(calibration_matrix), R, t);
    cv::Affine3d true_pose(R, cv::Vec3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));
    return true_pose;
}

cv::Affine3d find_pose_pnp(const std::vector<cv::Vec3d>& coords, const cv::Matx33d& calibration_matrix, const cv::Affine3d& pose) {
    std::vector<cv::Vec3d> camera_coords = make_camera_coords(coords, pose);
    std::vector<cv::Vec2d> flat_coords = make_3d_to_2d(camera_coords);
    std::vector<cv::Vec3d> coords3 = make_3_coord(flat_coords);
    std::vector<cv::Vec3d> pixel_coords = make_pixels(coords3, calibration_matrix);
    std::vector<cv::Vec2d> norm_pixel_coords = make_2_coord(pixel_coords);
    cv::Vec3d rvec, tvec;
    cv::solvePnP(coords, norm_pixel_coords, calibration_matrix, cv::Mat::zeros(4, 1, CV_64F), rvec, tvec);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Affine3d true_pose(R, tvec);
    return true_pose;

}

cv::Matx33d gramm_shmidt(const cv::Mat& R) {
    cv::Vec3d row0 = R.row(0);
    cv::Vec3d row1 = R.row(1);
    cv::Vec3d row2 = R.row(2);
    cv::Vec3d normed_row0 = cv::normalize(row0);
    cv::Vec3d proj1 = (row1.dot(normed_row0) / normed_row0.dot(normed_row0)) * normed_row0;
    cv::Vec3d orto_vector1 = row1 - proj1;
    cv::Vec3d normed_row1 = cv::normalize(orto_vector1);
    cv::Vec3d proj2_1 = (row2.dot(normed_row1) / normed_row1.dot(normed_row1)) * normed_row1;
    cv::Vec3d proj2_0 = (row2.dot(normed_row0) / normed_row0.dot(normed_row0)) * normed_row0;
    cv::Vec3d orto_vector2 = row2 - proj2_1 - proj2_0;
    cv::Vec3d normed_row2 = cv::normalize(orto_vector2);
    cv::Matx33d normed_R(
    normed_row0[0], normed_row0[1], normed_row0[2],
    normed_row1[0], normed_row1[1], normed_row1[2],
    normed_row2[0], normed_row2[1], normed_row2[2]);
    return normed_R;
}

int main() {
    cv::Vec3d rvec_2(0.1, -0.2, 0.3), t(1.1, -2.1, 3.);
    cv::Affine3d P_2(rvec_2, t);
    double fx = 300, fy = 300, cx = 320, cy = 320;
    cv::Matx33d cameraMatrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    std::vector<cv::Vec3d> X_world_points = {
        {3, 2, 5},
        {10, -3, 5.5},
        {-3, 1, 4.5},
        {8, 2, 4},
        {-2, -3, 4},
        {8, 3, 10},
        {-5, 4, 6},
        {2, -6, 7}
    };
    cv::Mat E = find_essential_matrix(X_world_points, cameraMatrix, P_2);
    cv::Mat F = find_fundamental_matrix(X_world_points, cameraMatrix, P_2);
    cv::Affine3d pose_from_essential = find_pose(X_world_points, cameraMatrix, P_2);
    cv::Affine3d pose_from_pnp = find_pose_pnp(X_world_points, cameraMatrix, P_2);
    cv::Matx33d R_essential_gs = gramm_shmidt(cv::Mat(pose_from_essential.rotation()));
    cv::Matx33d R_pnp_gs = gramm_shmidt(cv::Mat(pose_from_pnp.rotation()));
    cv::Affine3d pose_from_essential_gs(R_essential_gs, pose_from_essential.translation());
    cv::Affine3d pose_from_pnp_gs(R_pnp_gs, pose_from_pnp.translation());
    std::cout << "E =\n" << E << "\n\n";
    std::cout << "F =\n" << F << "\n\n";
    std::cout << "real R =\n" << cv::Mat(P_2.rotation()) << "\n";
    std::cout << "real t = " << P_2.translation() << "\n\n";
    std::cout << "R from essential =\n" << cv::Mat(pose_from_essential.rotation()) << "\n";
    std::cout << "t from essential = " << pose_from_essential.translation() << "\n\n";
    std::cout << "R from essential after Gram-Schmidt =\n" << cv::Mat(pose_from_essential_gs.rotation()) << "\n";
    std::cout << "t from essential after Gram-Schmidt = " << pose_from_essential_gs.translation() << "\n\n";
    std::cout << "R from PnP =\n" << cv::Mat(pose_from_pnp.rotation()) << "\n";
    std::cout << "t from PnP = " << pose_from_pnp.translation() << "\n\n";
    std::cout << "R from PnP after Gram-Schmidt =\n" << cv::Mat(pose_from_pnp_gs.rotation()) << "\n";
    std::cout << "t from PnP after Gram-Schmidt = " << pose_from_pnp_gs.translation() << "\n\n";

}





