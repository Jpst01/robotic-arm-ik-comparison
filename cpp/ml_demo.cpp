#include <iostream>
#include <iomanip>
#include <vector>

#include "kinematics/Kinematics.h"
#include "control/ControlLoop.h"
#include "ml_inference/MLInference.h"

using Eigen::Vector3d;

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║  3-DOF Robotic Arm — ML Inference Demo   ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    std::string model_path = "../models/ik_model.onnx";
    if (argc > 1) model_path = argv[1];

    ik::MLInference ml;
    if (!ml.loadModel(model_path)) {
        std::cerr << "Failed to load model from: " << model_path << "\n";
        std::cerr << "Usage: " << argv[0] << " [path/to/ik_model.onnx]\n";
        return 1;
    }

    ik::ControlLoop loop;
    ik::Kinematics kin;

    loop.setIKSolver([&](const Eigen::Vector3d& target) -> std::optional<Eigen::Vector3d> {
        auto ml_result = ml.predict(target);
        if (ml_result.has_value()) {
            Vector3d fk_pos = kin.forward(ml_result.value());
            double err = (fk_pos - target).norm();
            if (err < 0.05) {
                return ml_result;
            }
            std::cout << "  [ML fallback] prediction error " << err
                      << " > threshold, using analytical IK\n";
        }
        return kin.inverse(target);
    });

    std::vector<Vector3d> targets = {
        {0.25,  0.0,  0.45},
        {0.15,  0.15, 0.40},
        {0.0,   0.30, 0.35},
        {-0.1,  0.2,  0.50},
        {0.20, -0.10, 0.38},
        {0.30,  0.0,  0.30},
        {0.10,  0.10, 0.55},
    };

    double total_time = 0.0;
    int successes = 0;

    std::cout << std::fixed << std::setprecision(4);

    for (const auto& target : targets) {
        auto result = loop.moveTo(target);
        total_time += result.ik_solve_time_ms;
        if (result.success) ++successes;

        std::cout << "Target [" << target.transpose() << "]: "
                  << (result.success ? "OK" : "FAIL")
                  << "  IK: " << result.ik_solve_time_ms << " ms";
        if (result.success) {
            double pos_err = (result.final_position - target).norm();
            std::cout << "  pos_err: " << pos_err;
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Summary ===\n"
              << "Success: " << successes << "/" << targets.size() << "\n"
              << "Total IK time: " << total_time << " ms\n"
              << "Avg IK time: " << total_time / targets.size() << " ms\n";

    return 0;
}
