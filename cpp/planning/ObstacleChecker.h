#pragma once

#include <Eigen/Dense>
#include <vector>
#include "../kinematics/Kinematics.h"

namespace ik {

struct SphereObstacle {
    Eigen::Vector3d center;
    double radius;
};

class ObstacleChecker {
public:
    ObstacleChecker() = default;

    void addObstacle(const SphereObstacle& obs);

    void clearObstacles();

    bool checkCollision(const Eigen::Vector3d& q, const Kinematics& kin) const;

    int checkTrajectory(const std::vector<Eigen::Vector3d>& trajectory,
                        const Kinematics& kin) const;

    const std::vector<SphereObstacle>& obstacles() const { return obstacles_; }

private:
    std::vector<SphereObstacle> obstacles_;

    static double pointSegmentDistance(const Eigen::Vector3d& p,
                                       const Eigen::Vector3d& a,
                                       const Eigen::Vector3d& b);
};

}

