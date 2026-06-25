#include "puct_neural.hpp"

#include <ATen/ATen.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kCellFeatureCount = 4;
constexpr int kFeatureCount = kCellCount * kCellFeatureCount + kMaxActions + 4;
constexpr int kLegalMaskOffset = kCellCount * kCellFeatureCount;
constexpr int kScalarOffset = kLegalMaskOffset + kMaxActions;
std::mutex loader_mutex;
std::mutex mps_inference_mutex;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

torch::inductor::AOTIModelPackageLoader make_loader(const std::string& package_path) {
    std::lock_guard<std::mutex> lock(loader_mutex);
    return torch::inductor::AOTIModelPackageLoader(package_path, "model", false, 1, -1);
}

void write_state_features(const TrackedState& state, float* features) {
    std::fill(features, features + kFeatureCount, 0.0F);
    int opponent = other_player(state.current_player);

    for (int cell = 0; cell < kCellCount; ++cell) {
        int owner = state.owners[cell];
        features[cell] = owner == kEmpty ? 1.0F : 0.0F;
        features[kCellCount + cell] = owner == state.current_player ? 1.0F : 0.0F;
        features[2 * kCellCount + cell] = owner == opponent ? 1.0F : 0.0F;
    }

    for (std::size_t index = 0; index < state.selected.size(); ++index) {
        features[3 * kCellCount + state.selected[index]] = 1.0F;
    }

    ActionList legal_actions = state.legal_actions();
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        features[kLegalMaskOffset + legal_actions[index]] = 1.0F;
    }

    features[kScalarOffset] = static_cast<float>(state.selected.size()) / 3.0F;
    features[kScalarOffset + 1] = static_cast<float>(state.max_claims) / 3.0F;
    features[kScalarOffset + 2] =
        static_cast<float>(state.turn_start_largest) / static_cast<float>(kCellCount);
    features[kScalarOffset + 3] = state.opening_turn ? 1.0F : 0.0F;
}

std::vector<NeuralEvaluation> run_aoti_model_batch(
    torch::inductor::AOTIModelPackageLoader& loader,
    const std::vector<TrackedState>& states,
    int input_rows,
    NeuralDevice device,
    NeuralBatchTiming* timing) {
    if (states.empty()) {
        return {};
    }
    if (input_rows < static_cast<int>(states.size())) {
        throw std::runtime_error("neural batch has more states than input rows");
    }

    auto feature_started = Clock::now();
    std::vector<float> features(static_cast<std::size_t>(input_rows) * kFeatureCount, 0.0F);
    std::vector<ActionList> legal_actions(states.size());
    for (std::size_t index = 0; index < states.size(); ++index) {
        write_state_features(states[index], features.data() + index * kFeatureCount);
        legal_actions[index] = states[index].legal_actions();
    }
    for (int index = static_cast<int>(states.size()); index < input_rows; ++index) {
        std::copy(
            features.data(),
            features.data() + kFeatureCount,
            features.data() + static_cast<std::size_t>(index) * kFeatureCount);
    }
    auto feature_finished = Clock::now();

    at::Tensor policy;
    at::Tensor value_tensor;
    Clock::time_point input_started;
    Clock::time_point input_finished;
    Clock::time_point inference_started;
    Clock::time_point inference_finished;
    Clock::time_point output_started;
    Clock::time_point output_finished;
    auto run_inference = [&]() {
        input_started = Clock::now();
        auto input = at::from_blob(
                         features.data(),
                         {input_rows, kFeatureCount},
                         at::TensorOptions().dtype(at::kFloat))
                         .clone();
        if (device == NeuralDevice::Mps) {
            input = input.to(at::kMPS);
        }
        input_finished = Clock::now();

        inference_started = Clock::now();
        std::vector<at::Tensor> outputs = loader.run({input});
        if (device == NeuralDevice::Mps) {
            at::detail::getMPSHooks().deviceSynchronize();
        }
        inference_finished = Clock::now();
        if (outputs.size() != 2) {
            throw std::runtime_error("neural model must return policy logits and value");
        }

        output_started = Clock::now();
        if (device == NeuralDevice::Mps) {
            policy = outputs[0].to(at::kCPU).contiguous();
            value_tensor = outputs[1].to(at::kCPU).contiguous();
        } else {
            policy = outputs[0].contiguous();
            value_tensor = outputs[1].contiguous();
        }
        if (policy.size(0) < input_rows || value_tensor.size(0) < input_rows) {
            throw std::runtime_error("neural model returned fewer rows than the configured batch size");
        }
        output_finished = Clock::now();
    };
    if (device == NeuralDevice::Mps) {
        std::lock_guard<std::mutex> lock(mps_inference_mutex);
        run_inference();
    } else {
        run_inference();
    }

    auto postprocess_started = Clock::now();
    std::vector<NeuralEvaluation> evaluations = build_neural_evaluations(
        states,
        legal_actions,
        policy.data_ptr<float>(),
        value_tensor.data_ptr<float>());
    auto postprocess_finished = Clock::now();
    if (timing != nullptr) {
        timing->feature_ns = elapsed_ns(feature_started, feature_finished);
        timing->input_ns = elapsed_ns(input_started, input_finished);
        timing->inference_ns = elapsed_ns(inference_started, inference_finished);
        timing->output_ns = elapsed_ns(output_started, output_finished);
        timing->postprocess_ns = elapsed_ns(postprocess_started, postprocess_finished);
    }
    return evaluations;
}

class AotiBatchModel : public NeuralBatchModel {
public:
    AotiBatchModel(
        const std::string& package_path,
        int requested_batch_size,
        NeuralDevice requested_device)
        : loader_(make_loader(package_path)),
          batch_size_(requested_batch_size),
          device_(requested_device) {}

    std::vector<NeuralEvaluation> evaluate_batch(
        const std::vector<TrackedState>& states,
        NeuralBatchTiming* timing) override {
        return run_aoti_model_batch(loader_, states, batch_size_, device_, timing);
    }

private:
    torch::inductor::AOTIModelPackageLoader loader_;
    int batch_size_;
    NeuralDevice device_;
};

}  // namespace

std::unique_ptr<NeuralBatchModel> make_aoti_neural_batch_model(
    const std::string& package_path,
    int batch_size,
    NeuralDevice device) {
    return std::make_unique<AotiBatchModel>(package_path, batch_size, device);
}
