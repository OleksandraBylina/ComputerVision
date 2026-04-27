#include <iostream>
#include <ceres/ceres.h>

struct MyFunction {
    template <typename T>
    bool operator()(const T* const vars, T* residuals) const {
        const T& x = vars[0];
        const T& y = vars[1];
        const T& z = vars[2];

        residuals[0] = x + y - T(2);
        residuals[1] = T(2) * x - z + T(1);
        residuals[2] = y + z - T(3);

        return true;
    }
};

int main() {
    double vars[3] = {0.0, 0.0, 0.0};

    ceres::Problem problem;

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<MyFunction, 3, 3>(
            new MyFunction()
        ),
        nullptr,
        vars
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n\n";

    std::cout << "x = " << vars[0] << "\n";
    std::cout << "y = " << vars[1] << "\n";
    std::cout << "z = " << vars[2] << "\n";

    return 0;
}