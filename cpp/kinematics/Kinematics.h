#pragma once

#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <cmath>

namespace ik {

constexpr double L1 = 0.3;
constexpr double L2 = 0.25;
constexpr double L3 = 0.15;

constexpr double JOINT_MIN = -M_PI;
constexpr double JOINT_MAX =  M_PI;

constexpr double WORKSPACE_RADIUS = L2 + L3;

enum class ElbowConfig { UP, DOWN };

class Kinematics {
public:
    using JointAngles = Eigen::Vector3d;
    using Position    = Eigen::Vector3d;

    Kinematics() = default;

    Position forward(const JointAngles& q) const;

    std::optional<JointAngles> inverse(const Position& target,
                                        ElbowConfig config = ElbowConfig::UP) const;

    std::vector<JointAngles> inverseAll(const Position& target) const;

    std::vector<Position> linkPositions(const JointAngles& q) const;

    static bool withinLimits(const JointAngles& q);

    static JointAngles clampToLimits(const JointAngles& q);
};

}

