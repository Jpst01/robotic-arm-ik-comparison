#include "ObstacleChecker.h"
#include <algorithm>

namespace ik {

void ObstacleChecker::addObstacle(const SphereObstacle& obs) {
    obstacles_.push_back(obs);
}

void ObstacleChecker::clearObstacles() {
    obstacles_.clear();
}

bool ObstacleChecker::checkCollision(const Eigen::Vector3d& q,
                                      const Kinematics& kin) const {
    auto links = kin.linkPositions(q);

    for (size_t i = 0; i + 1 < links.size(); ++i) {
        const auto& a = links[i];
        const auto& b = links[i + 1];

        for (const auto& obs : obstacles_) {
            double dist = pointSegmentDistance(obs.center, a, b);
            if (dist < obs.radius) {
                return true;
            }
        }
    }
    return false;
}

int ObstacleChecker::checkTrajectory(
    const std::vector<Eigen::Vector3d>& trajectory,
    const Kinematics& kin) const {

    for (size_t i = 0; i < trajectory.size(); ++i) {
        if (checkCollision(trajectory[i], kin)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

double ObstacleChecker::pointSegmentDistance(const Eigen::Vector3d& p,
                                              const Eigen::Vector3d& a,
                                              const Eigen::Vector3d& b) {
    Eigen::Vector3d ab = b - a;
    Eigen::Vector3d ap = p - a;

    double ab_len2 = ab.squaredNorm();
    if (ab_len2 < 1e-12) {
        return ap.norm();
    }

    double t = std::clamp(ap.dot(ab) / ab_len2, 0.0, 1.0);
    Eigen::Vector3d closest = a + t * ab;

    return (p - closest).norm();
}

}

