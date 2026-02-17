#include "Kinematics.h"
#include <cmath>
#include <algorithm>

namespace ik {

Kinematics::Position Kinematics::forward(const JointAngles& q) const {
    double t1 = q[0], t2 = q[1], t3 = q[2];

    double r = L2 * std::cos(t2) + L3 * std::cos(t2 + t3);

    double x = std::cos(t1) * r;
    double y = std::sin(t1) * r;
    double z = L1 + L2 * std::sin(t2) + L3 * std::sin(t2 + t3);

    return {x, y, z};
}

std::optional<Kinematics::JointAngles>
Kinematics::inverse(const Position& target, ElbowConfig config) const {
    double x = target[0], y = target[1], z = target[2];

    double t1 = std::atan2(y, x);

    double r  = std::sqrt(x * x + y * y);
    double zp = z - L1;

    double d2 = r * r + zp * zp;
    double d  = std::sqrt(d2);

    if (d > L2 + L3 || d < std::fabs(L2 - L3)) {
        return std::nullopt;
    }

    double cos_t3 = (d2 - L2 * L2 - L3 * L3) / (2.0 * L2 * L3);
    cos_t3 = std::clamp(cos_t3, -1.0, 1.0);
    double t3 = std::acos(cos_t3);

    if (config == ElbowConfig::DOWN) {
        t3 = -t3;
    }

    double alpha = std::atan2(zp, r);
    double beta  = std::atan2(L3 * std::sin(t3), L2 + L3 * std::cos(t3));
    double t2    = alpha - beta;

    JointAngles q(t1, t2, t3);

    if (!withinLimits(q)) {
        q = clampToLimits(q);
        Position check = forward(q);
        if ((check - target).norm() > 0.01) {
            return std::nullopt;
        }
    }

    return q;
}

std::vector<Kinematics::JointAngles>
Kinematics::inverseAll(const Position& target) const {
    std::vector<JointAngles> solutions;
    auto up = inverse(target, ElbowConfig::UP);
    if (up.has_value()) solutions.push_back(up.value());
    auto down = inverse(target, ElbowConfig::DOWN);
    if (down.has_value()) solutions.push_back(down.value());
    return solutions;
}

std::vector<Kinematics::Position>
Kinematics::linkPositions(const JointAngles& q) const {
    double t1 = q[0], t2 = q[1], t3 = q[2];

    Position base(0.0, 0.0, 0.0);

    Position shoulder(0.0, 0.0, L1);

    double r_elbow = L2 * std::cos(t2);
    Position elbow(
        std::cos(t1) * r_elbow,
        std::sin(t1) * r_elbow,
        L1 + L2 * std::sin(t2)
    );

    Position ee = forward(q);

    return {base, shoulder, elbow, ee};
}

bool Kinematics::withinLimits(const JointAngles& q) {
    for (int i = 0; i < 3; ++i) {
        if (q[i] < JOINT_MIN || q[i] > JOINT_MAX) return false;
    }
    return true;
}

Kinematics::JointAngles Kinematics::clampToLimits(const JointAngles& q) {
    JointAngles clamped;
    for (int i = 0; i < 3; ++i) {
        clamped[i] = std::clamp(q[i], JOINT_MIN, JOINT_MAX);
    }
    return clamped;
}

}

