#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <functional>
#include "../kinematics/Kinematics.h"
#include "../planning/TrajectoryPlanner.h"
#include "../planning/ObstacleChecker.h"

namespace ik {

class ControlLoop {
public:
    using IKSolver = std::function<std::optional<Eigen::Vector3d>(const Eigen::Vector3d&)>;

    struct MoveResult {
        bool success = false;
        std::string message;
        std::vector<Eigen::Vector3d> trajectory;
        Eigen::Vector3d final_position;
        double ik_solve_time_ms = 0.0;
    };

    ControlLoop();

    MoveResult moveTo(const Eigen::Vector3d& target);

    void setCurrentAngles(const Eigen::Vector3d& q) { current_q_ = q; }
    Eigen::Vector3d currentAngles() const { return current_q_; }

    Kinematics& kinematics() { return kin_; }
    TrajectoryPlanner& planner() { return planner_; }
    ObstacleChecker& obstacleChecker() { return obstacle_checker_; }

    void setIKSolver(IKSolver solver) { ik_solver_ = std::move(solver); }

private:
    Kinematics kin_;
    TrajectoryPlanner planner_;
    ObstacleChecker obstacle_checker_;
    Eigen::Vector3d current_q_;
    IKSolver ik_solver_;

    std::optional<Eigen::Vector3d> defaultIK(const Eigen::Vector3d& target);

    std::optional<Eigen::Vector3d> tryReplan(const Eigen::Vector3d& target);
};

}

