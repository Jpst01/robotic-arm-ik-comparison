#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>

#include "../cpp/kinematics/Kinematics.h"
#include "../cpp/planning/TrajectoryPlanner.h"

using Eigen::Vector3d;

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        std::cout << "TEST: " << name << " ... "; \
    } while (0)

#define PASS() \
    do { \
        tests_passed++; \
        std::cout << "PASSED\n"; \
    } while (0)

#define FAIL(msg) \
    do { \
        std::cout << "FAILED: " << msg << "\n"; \
    } while (0)

void test_fk_zero() {
    TEST("FK at zero angles");
    ik::Kinematics kin;
    Vector3d q(0.0, 0.0, 0.0);
    Vector3d pos = kin.forward(q);

    double tol = 1e-10;
    if (std::abs(pos[0] - 0.4) < tol &&
        std::abs(pos[1] - 0.0) < tol &&
        std::abs(pos[2] - 0.3) < tol) {
        PASS();
    } else {
        FAIL("Expected (0.4, 0.0, 0.3), got (" +
             std::to_string(pos[0]) + ", " +
             std::to_string(pos[1]) + ", " +
             std::to_string(pos[2]) + ")");
    }
}

void test_fk_known() {
    TEST("FK at known angles");
    ik::Kinematics kin;
    Vector3d q(0.0, M_PI / 2, 0.0);
    Vector3d pos = kin.forward(q);

    double tol = 1e-10;
    if (std::abs(pos[0]) < tol &&
        std::abs(pos[1]) < tol &&
        std::abs(pos[2] - 0.7) < tol) {
        PASS();
    } else {
        FAIL("Expected (0, 0, 0.7), got (" +
             std::to_string(pos[0]) + ", " +
             std::to_string(pos[1]) + ", " +
             std::to_string(pos[2]) + ")");
    }
}

void test_ik_roundtrip() {
    TEST("IK round-trip (10 random positions)");
    ik::Kinematics kin;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> angle_dist(-1.0, 1.0);

    bool all_ok = true;
    for (int i = 0; i < 10; ++i) {
        Vector3d q_orig(angle_dist(rng), angle_dist(rng) * 0.8, std::abs(angle_dist(rng)));
        Vector3d target = kin.forward(q_orig);

        auto ik_result = kin.inverse(target);
        if (!ik_result.has_value()) {
            FAIL("IK returned nullopt for reachable target");
            all_ok = false;
            break;
        }

        Vector3d pos = kin.forward(ik_result.value());
        double err = (pos - target).norm();
        if (err > 0.01) {
            FAIL("Roundtrip error " + std::to_string(err) + " > 0.01");
            all_ok = false;
            break;
        }
    }
    if (all_ok) PASS();
}

void test_ik_unreachable() {
    TEST("IK unreachable target");
    ik::Kinematics kin;
    Vector3d target(5.0, 5.0, 5.0);
    auto result = kin.inverse(target);
    if (!result.has_value()) {
        PASS();
    } else {
        FAIL("Expected nullopt for unreachable target");
    }
}

void test_trajectory_continuity() {
    TEST("Trajectory continuity");
    ik::TrajectoryPlanner planner(1.0, 0.02);
    Vector3d start(0.0, 0.0, 0.0);
    Vector3d goal(1.0, 0.5, 0.8);

    auto traj = planner.plan(start, goal);

    if (traj.size() < 2) {
        FAIL("Trajectory too short");
        return;
    }

    bool ok = true;
    double max_step = 0.0;
    for (size_t i = 1; i < traj.size(); ++i) {
        double step = (traj[i] - traj[i - 1]).norm();
        max_step = std::max(max_step, step);
    }

    if (max_step > 0.1) {
        FAIL("Max step " + std::to_string(max_step) + " > 0.1 rad");
    } else {
        PASS();
    }
}

void test_trajectory_endpoints() {
    TEST("Trajectory start/end match");
    ik::TrajectoryPlanner planner(1.0, 0.02);
    Vector3d start(0.1, -0.2, 0.3);
    Vector3d goal(0.8, 0.4, -0.1);

    auto traj = planner.plan(start, goal);

    double tol = 1e-10;
    bool start_ok = (traj.front() - start).norm() < tol;
    bool end_ok = (traj.back() - goal).norm() < tol;

    if (start_ok && end_ok) {
        PASS();
    } else {
        FAIL("Endpoints don't match start/goal");
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║  Robotic Arm IK — Unit Tests             ║\n"
              << "╚══════════════════════════════════════════╝\n\n";

    test_fk_zero();
    test_fk_known();
    test_ik_roundtrip();
    test_ik_unreachable();
    test_trajectory_continuity();
    test_trajectory_endpoints();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run
              << " passed ===\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
