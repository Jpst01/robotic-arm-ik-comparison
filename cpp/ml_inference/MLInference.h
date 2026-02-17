#pragma once

#include <Eigen/Dense>
#include <string>
#include <optional>
#include <memory>

namespace Ort { class Session; class Env; class MemoryInfo; }

namespace ik {

class MLInference {
public:
    MLInference();
    ~MLInference();

    bool loadModel(const std::string& model_path);

    std::optional<Eigen::Vector3d> predict(const Eigen::Vector3d& target);

    bool isLoaded() const { return loaded_; }

private:
    bool loaded_ = false;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

}

