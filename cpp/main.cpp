#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#include "kinematics/Kinematics.h"
#include "planning/TrajectoryPlanner.h"
#include "planning/ObstacleChecker.h"
#include "control/ControlLoop.h"

using Eigen::Vector3d;

static void printHeader(const std::string& title) {
    std::cout << "\n========================================\n"
              << "  " << title
              << "\n========================================\n";
}

static void demoKinematics() {
    printHeader("Forward & Inverse Kinematics");
    ik::Kinematics kin;

    std::vector<Vector3d> test_angles = {
        {0.0, 0.5, 0.5},
        {M_PI / 4, 0.3, 0.8},
        {-M_PI / 3, 0.7, 0.2},
        {0.0, 0.0, 0.0},
    };

    for (const auto& q : test_angles) {
        Vector3d pos = kin.forward(q);
        auto ik_result = kin.inverse(pos);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Joints: [" << q.transpose() << "]  -->  "
                  << "Pos: [" << pos.transpose() << "]";

        if (ik_result) {
            Vector3d pos2 = kin.forward(ik_result.value());
            double err = (pos - pos2).norm();
            std::cout << "  -->  IK err: " << err << "\n";
        } else {
            std::cout << "  -->  IK: UNREACHABLE\n";
        }
    }
}

static void demoTrajectory() {
    printHeader("Trajectory Planning");
    ik::Kinematics kin;
    ik::TrajectoryPlanner planner(1.0, 0.02);

    Vector3d start(0.0, 0.0, 0.0);
    Vector3d goal(0.5, 0.8, 0.3);

    auto traj = planner.plan(start, goal);
    std::cout << "Planned " << traj.size() << " waypoints.\n";
    std::cout << "Start angles: [" << traj.front().transpose() << "]\n";
    std::cout << "Goal  angles: [" << traj.back().transpose() << "]\n";

    std::cout << "First 3 waypoints:\n";
    for (size_t i = 0; i < std::min<size_t>(3, traj.size()); ++i) {
        Vector3d p = kin.forward(traj[i]);
        std::cout << "  t=" << i << "  q=[" << traj[i].transpose()
                  << "]  pos=[" << p.transpose() << "]\n";
    }
    std::cout << "Last waypoint:\n";
    Vector3d p = kin.forward(traj.back());
    std::cout << "  t=" << traj.size() - 1 << "  q=[" << traj.back().transpose()
              << "]  pos=[" << p.transpose() << "]\n";
}

static void demoObstacles() {
    printHeader("Obstacle Avoidance");
    ik::ControlLoop loop;

    ik::SphereObstacle obs;
    obs.center = Vector3d(0.15, 0.0, 0.35);
    obs.radius = 0.05;
    loop.obstacleChecker().addObstacle(obs);

    std::cout << "Obstacle at [" << obs.center.transpose()
              << "] r=" << obs.radius << "\n\n";

    std::vector<Vector3d> targets = {
        {0.2, 0.0, 0.4},
        {0.3, 0.1, 0.35},
        {0.1, 0.2, 0.5},
    };

    for (const auto& target : targets) {
        auto result = loop.moveTo(target);
        std::cout << "Target [" << target.transpose() << "]: "
                  << result.message
                  << "  (IK time: " << result.ik_solve_time_ms << " ms)\n";
        if (result.success) {
            std::cout << "  Final pos: [" << result.final_position.transpose()
                      << "]  Waypoints: " << result.trajectory.size() << "\n";
        }
    }
}

static void demoControlLoop() {
    printHeader("Control Loop — Multi-target Sequence");
    ik::ControlLoop loop;
    loop.planner().setMaxJointVelocity(2.0);

    std::vector<Vector3d> targets = {
        {0.25,  0.0,  0.45},
        {0.15,  0.15, 0.40},
        {0.0,   0.30, 0.35},
        {-0.1,  0.2,  0.50},
        {0.20, -0.10, 0.38},
    };

    double total_ik_time = 0.0;
    int successes = 0;

    for (const auto& target : targets) {
        auto result = loop.moveTo(target);
        total_ik_time += result.ik_solve_time_ms;
        if (result.success) ++successes;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Target [" << target.transpose() << "]: "
                  << (result.success ? "OK" : "FAIL")
                  << "  IK: " << result.ik_solve_time_ms << " ms";
        if (result.success) {
            double pos_err = (result.final_position - target).norm();
            std::cout << "  pos_err: " << pos_err;
        }
        std::cout << "\n";
    }

    std::cout << "\nSummary: " << successes << "/" << targets.size()
              << " targets reached.  Total IK time: " << total_ik_time << " ms\n";
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n"
              << "║  3-DOF Robotic Arm — Algorithmic Demo    ║\n"
              << "╚══════════════════════════════════════════╝\n";

    demoKinematics();
    demoTrajectory();
    demoObstacles();
    demoControlLoop();

    return 0;
}
