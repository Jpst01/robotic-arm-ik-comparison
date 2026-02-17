#pragma once

#include <Eigen/Dense>
#include <vector>

namespace ik {

class TrajectoryPlanner {
public:
    using JointAngles = Eigen::Vector3d;

    explicit TrajectoryPlanner(double max_joint_vel = 1.0, double dt = 0.01);

    std::vector<JointAngles> plan(const JointAngles& start,
                                  const JointAngles& goal) const;

    void setMaxJointVelocity(double v) { max_joint_vel_ = v; }
    void setTimestep(double dt) { dt_ = dt; }

private:
    double max_joint_vel_;
    double dt_;
};

}

