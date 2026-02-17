#include "TrajectoryPlanner.h"
#include <cmath>
#include <algorithm>

namespace ik {

TrajectoryPlanner::TrajectoryPlanner(double max_joint_vel, double dt)
    : max_joint_vel_(max_joint_vel), dt_(dt) {}

std::vector<TrajectoryPlanner::JointAngles>
TrajectoryPlanner::plan(const JointAngles& start, const JointAngles& goal) const {
    std::vector<JointAngles> trajectory;

    JointAngles diff = goal - start;
    double max_displacement = diff.cwiseAbs().maxCoeff();

    if (max_displacement < 1e-8) {
        trajectory.push_back(start);
        return trajectory;
    }

    double duration = max_displacement / max_joint_vel_;

    int steps = std::max(1, static_cast<int>(std::ceil(duration / dt_)));

    trajectory.reserve(steps + 1);

    for (int i = 0; i <= steps; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(steps);

        double s = 3.0 * t * t - 2.0 * t * t * t;

        JointAngles waypoint = start + s * diff;
        trajectory.push_back(waypoint);
    }

    return trajectory;
}

}

