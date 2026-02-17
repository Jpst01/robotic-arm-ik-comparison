#include "MLInference.h"
#include "../kinematics/Kinematics.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <array>

namespace ik {

MLInference::MLInference() = default;
MLInference::~MLInference() = default;

bool MLInference::loadModel(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ik_ml");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);
        loaded_ = true;
        std::cout << "[MLInference] Model loaded: " << model_path << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[MLInference] Failed to load model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::optional<Eigen::Vector3d> MLInference::predict(const Eigen::Vector3d& target) {
    if (!loaded_) return std::nullopt;

    try {
        std::array<float, 3> input_data = {
            static_cast<float>(target[0]),
            static_cast<float>(target[1]),
            static_cast<float>(target[2])
        };
        std::array<int64_t, 2> input_shape = {1, 3};

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        const char* input_names[]  = {"input"};
        const char* output_names[] = {"output"};

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        Eigen::Vector3d q(output_data[0], output_data[1], output_data[2]);

        if (!Kinematics::withinLimits(q)) {
            q = Kinematics::clampToLimits(q);
        }

        return q;
    } catch (const Ort::Exception& e) {
        std::cerr << "[MLInference] Inference error: " << e.what() << std::endl;
        return std::nullopt;
    }
}

}

