#include "ControlLoop.h"
#include <chrono>
#include <iostream>

namespace ik {

ControlLoop::ControlLoop()
    : current_q_(Eigen::Vector3d::Zero()) {
    ik_solver_ = [this](const Eigen::Vector3d& t) { return defaultIK(t); };
}

ControlLoop::MoveResult ControlLoop::moveTo(const Eigen::Vector3d& target) {
    MoveResult result;
    result.final_position = kin_.forward(current_q_);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto ik_result = ik_solver_(target);
    auto t_end = std::chrono::high_resolution_clock::now();

    result.ik_solve_time_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (!ik_result.has_value()) {
        result.success = false;
        result.message = "IK failed: target unreachable";
        return result;
    }

    Eigen::Vector3d goal_q = ik_result.value();

    auto trajectory = planner_.plan(current_q_, goal_q);

    int collision_idx = obstacle_checker_.checkTrajectory(trajectory, kin_);
    if (collision_idx >= 0) {
        auto replan = tryReplan(target);
        if (replan.has_value()) {
            goal_q = replan.value();
            trajectory = planner_.plan(current_q_, goal_q);

            collision_idx = obstacle_checker_.checkTrajectory(trajectory, kin_);
            if (collision_idx >= 0) {
                result.success = false;
                result.message = "Collision detected; replanning failed";
                return result;
            }
        } else {
            result.success = false;
            result.message = "Collision detected; no valid replan found";
            return result;
        }
    }

    result.trajectory = trajectory;
    current_q_ = trajectory.back();
    result.final_position = kin_.forward(current_q_);
    result.success = true;
    result.message = "Motion complete";

    return result;
}

std::optional<Eigen::Vector3d>
ControlLoop::defaultIK(const Eigen::Vector3d& target) {
    return kin_.inverse(target);
}

std::optional<Eigen::Vector3d>
ControlLoop::tryReplan(const Eigen::Vector3d& target) {
    const double offsets[] = {0.02, -0.02, 0.04, -0.04};

    for (int axis = 0; axis < 3; ++axis) {
        for (double off : offsets) {
            Eigen::Vector3d shifted = target;
            shifted[axis] += off;
            auto ik_result = ik_solver_(shifted);
            if (ik_result.has_value()) {
                auto traj = planner_.plan(current_q_, ik_result.value());
                if (obstacle_checker_.checkTrajectory(traj, kin_) < 0) {
                    return ik_result;
                }
            }
        }
    }
    return std::nullopt;
}

}

