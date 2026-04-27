#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>


struct CameraPointSet {
    double R[3];
    double t[3];
    double point[3];
};

struct PointResidual {
    PointResidual(double res_u, double res_v)
        : res_u_(res_u), res_v_(res_v) {}
    template <typename T>
    bool operator()(const T* const R,
                    const T* const t,
                    const T* const Point,
                    T* residual) const {
        T p[3];
        ceres::AngleAxisRotatePoint(R, Point, p);
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];
        T u = p[0] / p[2];
        T v = p[1] / p[2];
        residual[0] = u - T(res_u_);
        residual[1] = v - T(res_v_);

        return true;
    }

private:
    double res_u_;
    double res_v_;
};

void project(const double R[3], const double t[3],
             const double point[3], double& u, double& v) {
    double p[3];
    ceres::AngleAxisRotatePoint(R, point, p);
    p[0] += t[0];
    p[1] += t[1];
    p[2] += t[2];
    u = p[0] / p[2];
    v = p[1] / p[2];
}

int main() {
    const int N = 8;
    double true_points[N][3] = {
        {0,0,5}, {1,0.2,6}, {-1,0.5,5.5}, {0.5,-0.7,4.5},
        {-0.4,-0.6,7}, {1.2,-0.4,8}, {-1.4,0.8,6.5}, {0.3,1,7.5}
    };

    double R1[3] = {0,0,0};
    double t1[3] = {0,0,0};

    double R2_true[3] = {0.03, -0.08, 0.02};
    double t2_true[3] = {-1, 0, 0};

    double obs1[N][2];
    double obs2[N][2];

    for (int i = 0; i < N; ++i) {
        project(R1, t1, true_points[i], obs1[i][0], obs1[i][1]);
        project(R2_true, t2_true, true_points[i], obs2[i][0], obs2[i][1]);
    }

    double points[N][3];
    for (int i = 0; i < N; ++i) {
        points[i][0] = true_points[i][0] + 0.1;
        points[i][1] = true_points[i][1] - 0.1;
        points[i][2] = true_points[i][2] - 0.2;
    }

    double R2[3] = {0,0,0};
    double t2[3] = {-0.8, 0.1, 0.1};

    ceres::Problem problem;
    for (int i = 0; i < N; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointResidual, 2, 3, 3, 3>(
                new PointResidual(obs1[i][0], obs1[i][1])
            ),
            nullptr,
            R1,
            t1,
            points[i]
        );
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointResidual, 2, 3, 3, 3>(
                new PointResidual(obs2[i][0], obs2[i][1])
            ),
            nullptr,
            R2,
            t2,
            points[i]
        );
    }


    problem.SetParameterBlockConstant(R1);
    problem.SetParameterBlockConstant(t1);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "R2:\n" << R2[0] << " " << R2[1] << " " << R2[2] << "\n";
    std::cout << "t2:\n" << t2[0] << " " << t2[1] << " " << t2[2] << "\n";

    std::cout << "\nPoints:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << i << ": "
                  << points[i][0] << " "
                  << points[i][1] << " "
                  << points[i][2] << "\n";
    }
}