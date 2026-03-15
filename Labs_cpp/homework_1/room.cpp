#include "room.h"
#include <opencv2/opencv.hpp>
#include <string>

void loadImageEx()
{
	std::string path = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/raven2.jpg";
	cv::Mat image = cv::imread(path);
	cv::imshow("Picture", image);
	cv::waitKey(0);
	cv::Mat grey_image = cv::imread(path, cv::IMREAD_GRAYSCALE);
	cv::imshow("Grey picture", grey_image);
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

void drawImage(const cv::Affine3d& P, const cv::Matx33d& cameraMatrix, const cv::Mat& wall)
{
	cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);
	for (int i = 0; i < wall.rows; ++i)
		for (int j = 0; j < wall.cols; ++j)
		{
			cv::Matx33d R = P.rotation();
			cv::Vec3d t = P.translation();

			cv::Vec3d X_world(i - wall.rows / 2, j - wall.cols / 2, 1000.);
			cv::Vec3d X_camera = R * X_world + t;
			cv::Vec3d proj(X_camera[0] / X_camera[2], X_camera[1] / X_camera[2]);
			cv::Vec3d pixels_projective = cameraMatrix * X_camera;
			cv::Vec2d pixels(pixels_projective[0] / pixels_projective[2], pixels_projective[1] / pixels_projective[2]);

			if (pixels[0] >= 0. && pixels[0] < 640. && pixels[1] >= 0. && pixels[1] < 640.)
				image.at<cv::Vec3b>(int(pixels[0]), int(pixels[1])) = wall.at<cv::Vec3b>(i, j);

		}

	cv::imshow("Transformed image", image);
	cv::waitKey(0);
}

void drawImage2(const cv::Affine3d& P, const cv::Matx33d& cameraMatrix, const cv::Mat& wall1, const cv::Mat& wall2, const cv::Mat& wall3, const cv::Mat& wall4, const cv::Mat& wall5)
{
	cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);
	cv::Matx33d R = P.rotation();
	cv::Vec3d t = P.translation();
	cv::Matx33d R_inv = R.inv();
	cv::Matx33d K_inv = cameraMatrix.inv();
	double eps = 1e-9;
	for (int i = 0; i < 640; ++i)
		for (int j = 0; j < 640; ++j)
		{


			cv::Vec3d pixels_projective(j, i, 1.);
			cv::Vec3d X_camera = K_inv * pixels_projective;

			std::vector<double> lengths(5, std::numeric_limits<double>::infinity());
			std::vector<cv::Vec2d> suitable(5, cv::Vec2d(-1., -1.));

			if (std::abs((R_inv * X_camera)[2]) > eps) {
				double k_backwall = ((R_inv * t)[2] + 1000.) / (R_inv * X_camera)[2];
				if (k_backwall > 0) {
					cv::Vec3d X_world_backwall = R_inv * (k_backwall * X_camera - t);
					if (X_world_backwall[0] >= -350. && X_world_backwall[0] <= 350. &&
						X_world_backwall[1] >= -300. && X_world_backwall[1] <= 300.) {
						cv::Vec2d pix_backwall(X_world_backwall[1] + wall1.rows / 2, X_world_backwall[0] + wall1.cols / 2);
						if (pix_backwall[0] >= 0. && pix_backwall[0] < wall1.rows && pix_backwall[1] >= 0. && pix_backwall[1] < wall1.cols) {
							lengths[0] = k_backwall;
							suitable[0] = pix_backwall;
						}
					}
				}
			}

			if (std::abs((R_inv * X_camera)[0]) > eps) {
				double k_leftwall = ((R_inv * t)[0] - 350.) / (R_inv * X_camera)[0];
				if (k_leftwall > 0) {
					cv::Vec3d X_world_leftwall = R_inv * (k_leftwall * X_camera - t);
					if (X_world_leftwall[2] >= 0. && X_world_leftwall[2] <= 1000. &&
						X_world_leftwall[1] >= -300. && X_world_leftwall[1] <= 300.) {
						cv::Vec2d pix_leftwall(X_world_leftwall[1] + wall2.rows / 2, X_world_leftwall[2]);
						if (pix_leftwall[0] >= 0. && pix_leftwall[0] < wall2.rows && pix_leftwall[1] >= 0. && pix_leftwall[1] < wall2.cols) {
							lengths[1] = k_leftwall;
							suitable[1] = pix_leftwall;
						}
					}
				}
			}

			if (std::abs((R_inv * X_camera)[0]) > eps) {
				double k_rightwall = ((R_inv * t)[0] + 350.) / (R_inv * X_camera)[0];
				if (k_rightwall > 0) {
					cv::Vec3d X_world_rightwall = R_inv * (k_rightwall * X_camera - t);
					if (X_world_rightwall[2] >= 0. && X_world_rightwall[2] <= 1000. &&
						X_world_rightwall[1] >= -300. && X_world_rightwall[1] <= 300.) {
						cv::Vec2d pix_rightwall(X_world_rightwall[1] + wall3.rows / 2, X_world_rightwall[2]);
						if (pix_rightwall[0] >= 0. && pix_rightwall[0] < wall3.rows && pix_rightwall[1] >= 0. && pix_rightwall[1] < wall3.cols) {
							lengths[2] = k_rightwall;
							suitable[2] = pix_rightwall;
						}
					}
				}
			}

			if (std::abs((R_inv * X_camera)[1]) > eps) {
				double k_floor = ((R_inv * t)[1] - 300.) / (R_inv * X_camera)[1];
				if (k_floor > 0) {
					cv::Vec3d X_world_floor = R_inv * (k_floor * X_camera - t);
					if (X_world_floor[0] >= -350. && X_world_floor[0] <= 350. &&
						X_world_floor[2] >= 0. && X_world_floor[2] <= 1000.) {
						cv::Vec2d pix_floor(X_world_floor[2], X_world_floor[0] + wall4.cols / 2);
						if (pix_floor[0] >= 0. && pix_floor[0] < wall4.rows && pix_floor[1] >= 0. && pix_floor[1] < wall4.cols) {
							lengths[3] = k_floor;
							suitable[3] = pix_floor;
						}
					}
				}
			}

			if (std::abs((R_inv * X_camera)[1]) > eps) {
				double k_ceiling = ((R_inv * t)[1] + 300.) / (R_inv * X_camera)[1];
				if (k_ceiling > 0) {
					cv::Vec3d X_world_ceiling = R_inv * (k_ceiling * X_camera - t);
					if (X_world_ceiling[0] >= -350. && X_world_ceiling[0] <= 350. &&
						X_world_ceiling[2] >= 0. && X_world_ceiling[2] <= 1000.) {
						cv::Vec2d pix_celing(X_world_ceiling[2], X_world_ceiling[0] + wall5.cols / 2);
						if (pix_celing[0] >= 0. && pix_celing[0] < wall5.rows && pix_celing[1] >= 0. && pix_celing[1] < wall5.cols) {
							lengths[4] = k_ceiling;
							suitable[4] = pix_celing;
						}
					}
				}
			}

			double min_k = std::numeric_limits<double>::infinity();
			cv::Vec2d winner(-1., -1.);
			int winner_idx = -1;
			bool found = false;
			for (int k = 0; k < 5; k++) {
				if (lengths[k] < min_k) {
					min_k = lengths[k];
					winner = suitable[k];
					winner_idx = k;
					found = true;
				}
			}
			if (found) {
				if (winner_idx == 0)
					image.at<cv::Vec3b>(i, j) = wall1.at<cv::Vec3b>(int(winner[0]), int(winner[1]));
				if (winner_idx == 1)
					image.at<cv::Vec3b>(i, j) = wall2.at<cv::Vec3b>(int(winner[0]), int(winner[1]));
				if (winner_idx == 2)
					image.at<cv::Vec3b>(i, j) = wall3.at<cv::Vec3b>(int(winner[0]), int(winner[1]));
				if (winner_idx == 3)
					image.at<cv::Vec3b>(i, j) = wall4.at<cv::Vec3b>(int(winner[0]), int(winner[1]));
				if (winner_idx == 4)
					image.at<cv::Vec3b>(i, j) = wall5.at<cv::Vec3b>(int(winner[0]), int(winner[1]));
			}

		}

	cv::imshow("Transformed image", image);
	cv::waitKey(0);
}

void useCameraTransform() {
	std::string path1 = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/jackdow.jpg";
	std::string path2 = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/raven.jpg";
	std::string path3 = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/magpie.jpg";
	std::string path4 = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/rook.jpg";
	std::string path5 = "/Users/oleksandrabylina/Documents/University/ComputerVision/Labs_cpp/homework_1/crow.jpg";
	cv::Mat wall1 = cv::imread(path1);
	cv::Mat wall2 = cv::imread(path2);
	cv::Mat wall3 = cv::imread(path3);
	cv::Mat wall4 = cv::imread(path4);
	cv::Mat wall5 = cv::imread(path5);



	cv::Vec3d rvec(0., 0., 0.), t(0., 0., 0.);
	cv::Affine3d P(rvec, t);

	double fx = 300, fy = 300, cx = 320, cy = 320;
	cv::Matx33d cameraMatrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);

	drawImage2(P, cameraMatrix, wall1, wall2, wall3, wall4, wall5);


}
