#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <string>

#include "kinematics/Kinematics.h"
#include "ml_inference/MLInference.h"

using Eigen::Vector3d;

struct Sample {
    Vector3d target;
    Vector3d ground_truth_q;
    double   analytical_time_ms = 0;
    double   analytical_pos_err = -1;
    double   analytical_angle_err = -1;
    bool     analytical_ok = false;
    double   ml_time_ms = 0;
    double   ml_pos_err = -1;
    double   ml_angle_err = -1;
    bool     ml_ok = false;
};

static long getMemoryKB() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            for (char c : line) {
                if (c >= '0' && c <= '9') kb = kb * 10 + (c - '0');
            }
            return kb;
        }
    }
    return -1;
}

struct Stats {
    int total = 0, success = 0;
    double total_time_ms = 0, total_pos_err = 0, max_pos_err = 0;
    double total_angle_err = 0, max_angle_err = 0;
};

static void printStats(const std::string& name, const Stats& s) {
    std::cout << std::fixed << std::setprecision(6);
    double avg_pos = s.success > 0 ? s.total_pos_err / s.success : 0;
    double avg_ang = s.success > 0 ? s.total_angle_err / s.success : 0;
    std::cout << "\n--- " << name << " ---\n"
              << "  Targets:           " << s.total << "\n"
              << "  Successes:         " << s.success
              << " (" << std::setprecision(1) << 100.0 * s.success / s.total << "%)\n"
              << std::setprecision(4)
              << "  Total time:        " << s.total_time_ms << " ms\n"
              << "  Avg time/solve:    " << s.total_time_ms / s.total << " ms\n"
              << "  Avg pos error:     " << std::setprecision(8) << avg_pos << " m\n"
              << "  Max pos error:     " << s.max_pos_err << " m\n"
              << "  Avg joint error:   " << std::setprecision(6) << avg_ang
              << " rad (" << std::setprecision(2) << avg_ang * 180.0 / M_PI << "°)\n"
              << "  Max joint error:   " << std::setprecision(6) << s.max_angle_err
              << " rad (" << std::setprecision(2) << s.max_angle_err * 180.0 / M_PI << "°)\n";
}

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║  IK Benchmark: Analytical vs. ML         ║\n"
              << "╚══════════════════════════════════════════╝\n";

    std::string model_path = "../models/ik_model.onnx";
    std::string csv_path   = "../data/benchmark_results.csv";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) csv_path   = argv[2];

    ik::Kinematics kin;
    long mem_before = getMemoryKB();

    ik::MLInference ml;
    bool ml_available = ml.loadModel(model_path);
    long mem_after_ml = getMemoryKB();

    if (!ml_available) {
        std::cerr << "[WARN] ML model not loaded — analytical only.\n";
    }

    const int N = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> angle_dist(-M_PI * 0.8, M_PI * 0.8);

    std::vector<Sample> samples(N);
    for (int i = 0; i < N; ++i) {
        Vector3d q(angle_dist(rng), angle_dist(rng) * 0.5, angle_dist(rng) * 0.5);
        samples[i].target = kin.forward(q);
        samples[i].ground_truth_q = q;
    }

    Stats analytical;
    analytical.total = N;

    for (auto& s : samples) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = kin.inverse(s.target);
        auto t1 = std::chrono::high_resolution_clock::now();

        s.analytical_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        analytical.total_time_ms += s.analytical_time_ms;

        if (result.has_value()) {
            s.analytical_ok = true;
            analytical.success++;
            Vector3d pos = kin.forward(result.value());
            s.analytical_pos_err = (pos - s.target).norm();
            s.analytical_angle_err = (result.value() - s.ground_truth_q).cwiseAbs().maxCoeff();
            analytical.total_pos_err += s.analytical_pos_err;
            analytical.max_pos_err = std::max(analytical.max_pos_err, s.analytical_pos_err);
            analytical.total_angle_err += s.analytical_angle_err;
            analytical.max_angle_err = std::max(analytical.max_angle_err, s.analytical_angle_err);
        }
    }

    printStats("Analytical IK", analytical);

    std::cout << "\n--- Multiple IK Configurations ---\n";
    Vector3d demo_target = samples[0].target;
    auto all_solutions = kin.inverseAll(demo_target);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Target: [" << demo_target.transpose() << "]\n";
    std::cout << "  Solutions found: " << all_solutions.size() << "\n";
    for (size_t i = 0; i < all_solutions.size(); ++i) {
        std::string label = (i == 0) ? "elbow-up" : "elbow-down";
        Vector3d pos = kin.forward(all_solutions[i]);
        double err = (pos - demo_target).norm();
        std::cout << "    " << label << ": ["
                  << all_solutions[i].transpose() << "]  pos_err=" << err << "\n";
    }

    Stats ml_stats;
    if (ml_available) {
        ml_stats.total = N;

        for (auto& s : samples) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto result = ml.predict(s.target);
            auto t1 = std::chrono::high_resolution_clock::now();

            s.ml_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ml_stats.total_time_ms += s.ml_time_ms;

            if (result.has_value()) {
                Vector3d pos = kin.forward(result.value());
                double pos_err = (pos - s.target).norm();
                double angle_err = (result.value() - s.ground_truth_q).cwiseAbs().maxCoeff();
                if (pos_err < 0.1) {
                    s.ml_ok = true;
                    s.ml_pos_err = pos_err;
                    s.ml_angle_err = angle_err;
                    ml_stats.success++;
                    ml_stats.total_pos_err += pos_err;
                    ml_stats.max_pos_err = std::max(ml_stats.max_pos_err, pos_err);
                    ml_stats.total_angle_err += angle_err;
                    ml_stats.max_angle_err = std::max(ml_stats.max_angle_err, angle_err);
                }
            }
        }

        printStats("ML IK (ONNX)", ml_stats);
    }

    std::cout << "\n--- Memory Footprint ---\n";
    if (mem_before > 0) {
        std::cout << "  Process RSS (before ML load): " << mem_before << " KB\n";
        std::cout << "  Process RSS (after  ML load): " << mem_after_ml << " KB\n";
        std::cout << "  ML model overhead:            " << (mem_after_ml - mem_before) << " KB\n";
    } else {
        std::cout << "  (Memory info not available on this platform)\n";
    }

    if (ml_available) {
        std::cout << "\n=== COMPARISON ===\n" << std::fixed
                  << "  Analytical success rate: " << std::setprecision(1)
                  << 100.0 * analytical.success / N << "%\n"
                  << "  ML success rate:         "
                  << 100.0 * ml_stats.success / N << "%\n"
                  << std::setprecision(4)
                  << "  Analytical avg time:     " << analytical.total_time_ms / N << " ms\n"
                  << "  ML avg time:             " << ml_stats.total_time_ms / N << " ms\n"
                  << "  Speedup (analytical/ML): "
                  << (ml_stats.total_time_ms > 0
                          ? analytical.total_time_ms / ml_stats.total_time_ms : 0.0)
                  << "x\n";
    }

    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "x,y,z,a_time_ms,a_pos_err,a_angle_err,a_ok,"
            << "ml_time_ms,ml_pos_err,ml_angle_err,ml_ok\n";
        csv << std::fixed << std::setprecision(8);
        for (const auto& s : samples) {
            csv << s.target[0] << "," << s.target[1] << "," << s.target[2] << ","
                << s.analytical_time_ms << "," << s.analytical_pos_err << ","
                << s.analytical_angle_err << "," << s.analytical_ok << ","
                << s.ml_time_ms << "," << s.ml_pos_err << ","
                << s.ml_angle_err << "," << s.ml_ok << "\n";
        }
        csv.close();
        std::cout << "\nResults exported to: " << csv_path << "\n";
    }

    return 0;
}
