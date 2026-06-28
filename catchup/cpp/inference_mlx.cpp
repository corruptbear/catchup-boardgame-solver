#include "puct_neural.hpp"

#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = mlx::core;

namespace {

constexpr int kCellFeatureCount = 4;
constexpr int kGlobalFeatureCount = 4;
constexpr int kMlxInputChannels = kCellFeatureCount + 1 + kGlobalFeatureCount + 1;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

mx::array array_from_values(const std::vector<float>& values, mx::Shape shape) {
    return mx::array(values.begin(), std::move(shape), mx::float32);
}

template <std::size_t N>
mx::array array_from_values(const std::array<float, N>& values, mx::Shape shape) {
    return mx::array(values.begin(), std::move(shape), mx::float32);
}

std::array<float, kCellCount * 6 * kCellCount> make_compact_direction_wide_values() {
    constexpr std::array<std::pair<int, int>, 6> directions = {{
        {1, 0},
        {-1, 0},
        {0, 1},
        {0, -1},
        {1, -1},
        {-1, 1},
    }};

    std::array<float, kCellCount * 6 * kCellCount> values{};
    const Board& current_board = board();
    for (int direction = 0; direction < 6; ++direction) {
        auto [dq, dr] = directions[direction];
        for (int destination = 0; destination < kCellCount; ++destination) {
            auto [q, r] = current_board.coords[destination];
            int source_q = q + dq;
            int source_r = r + dr;
            for (int source = 0; source < kCellCount; ++source) {
                if (current_board.coords[source] == std::pair<int, int>{source_q, source_r}) {
                    values[source * (6 * kCellCount) + direction * kCellCount + destination] = 1.0F;
                    break;
                }
            }
        }
    }
    return values;
}

mx::array linear(const mx::array& x, const mx::array& weight, const mx::array& bias) {
    return mx::matmul(x, mx::transpose(weight)) + bias;
}

mx::array relu(const mx::array& x) {
    return mx::maximum(x, mx::array(0.0F));
}

mx::array require_weight(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& name) {
    auto found = weights.find(name);
    if (found == weights.end()) {
        throw std::runtime_error("missing MLX weight: " + name);
    }
    return found->second;
}

class MlxDirectionalModel : public NeuralBatchModel {
public:
    MlxDirectionalModel(std::string weights_path, int requested_batch_size)
        : weights_(mx::load_safetensors(weights_path).first),
          batch_size_(requested_batch_size),
          dual_value_(weights_.find("win_value_head.0.weight") != weights_.end()),
          direction_wide_matrix_(array_from_values(
              make_compact_direction_wide_values(),
              {kCellCount, 6 * kCellCount})) {
        if (batch_size_ <= 0) {
            throw std::runtime_error("MLX batch size must be positive");
        }
        compiled_forward_ = mx::compile(
            std::function<std::vector<mx::array>(const std::vector<mx::array>&)>(
                [this](const std::vector<mx::array>& inputs) {
                    return forward(inputs);
                }),
            false);
        auto warmup_input = mx::zeros({batch_size_, kCellCount, kMlxInputChannels});
        auto warmup_global = mx::zeros({batch_size_, kGlobalFeatureCount + 1});
        auto outputs = compiled_forward_({warmup_input, warmup_global});
        mx::eval(outputs);
    }

    std::vector<NeuralEvaluation> evaluate_batch(
        const std::vector<TrackedState>& states,
        NeuralBatchTiming* timing) override {
        if (states.empty()) {
            return {};
        }
        if (static_cast<int>(states.size()) > batch_size_) {
            throw std::runtime_error("MLX neural batch has more states than input rows");
        }

        auto feature_started = Clock::now();
        std::vector<float> input_values(
            static_cast<std::size_t>(batch_size_) * kCellCount * kMlxInputChannels,
            0.0F);
        std::vector<float> global_values(
            static_cast<std::size_t>(batch_size_) * (kGlobalFeatureCount + 1),
            0.0F);
        std::vector<ActionList> legal_actions(states.size());

        for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
            const TrackedState& state = states[state_index];
            int opponent = other_player(state.current_player);
            legal_actions[state_index] = state.legal_actions();

            float finish_legal = 0.0F;
            std::array<float, kCellCount> legal_claims{};
            for (std::size_t action_index = 0; action_index < legal_actions[state_index].size(); ++action_index) {
                int action = legal_actions[state_index][action_index];
                if (action == kFinish) {
                    finish_legal = 1.0F;
                } else {
                    legal_claims[action] = 1.0F;
                }
            }

            float globals[kGlobalFeatureCount + 1] = {
                static_cast<float>(state.selected.size()) / 3.0F,
                static_cast<float>(state.max_claims) / 3.0F,
                static_cast<float>(state.turn_start_largest) / static_cast<float>(kCellCount),
                state.opening_turn ? 1.0F : 0.0F,
                finish_legal,
            };
            std::copy(
                globals,
                globals + kGlobalFeatureCount + 1,
                global_values.data() + state_index * (kGlobalFeatureCount + 1));

            for (int cell = 0; cell < kCellCount; ++cell) {
                float* features = input_values.data()
                    + (state_index * kCellCount + cell) * kMlxInputChannels;
                int owner = state.owners[cell];
                features[0] = owner == kEmpty ? 1.0F : 0.0F;
                features[1] = owner == state.current_player ? 1.0F : 0.0F;
                features[2] = owner == opponent ? 1.0F : 0.0F;
                features[4] = legal_claims[cell];
                for (int selected_index = 0; selected_index < static_cast<int>(state.selected.size()); ++selected_index) {
                    if (state.selected[selected_index] == cell) {
                        features[3] = 1.0F;
                        break;
                    }
                }
                for (int global_index = 0; global_index < kGlobalFeatureCount + 1; ++global_index) {
                    features[5 + global_index] = globals[global_index];
                }
            }
        }
        auto feature_finished = Clock::now();

        auto input_started = Clock::now();
        mx::array input_grid = array_from_values(
            input_values,
            {batch_size_, kCellCount, kMlxInputChannels});
        mx::array global_features = array_from_values(
            global_values,
            {batch_size_, kGlobalFeatureCount + 1});
        auto input_finished = Clock::now();

        auto inference_started = Clock::now();
        auto outputs = compiled_forward_({input_grid, global_features});
        mx::eval(outputs);
        auto inference_finished = Clock::now();

        auto output_started = Clock::now();
        const float* logits = outputs[0].data<float>();
        const float* win_values = outputs[1].data<float>();
        const float* margin_values = dual_value_ ? outputs[2].data<float>() : nullptr;
        auto output_finished = Clock::now();

        auto postprocess_started = Clock::now();
        std::vector<NeuralEvaluation> evaluations =
            build_neural_evaluations(states, legal_actions, logits, win_values, margin_values);
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

private:
    std::vector<mx::array> forward(const std::vector<mx::array>& inputs) const {
        mx::array hidden = relu(conv1x1(inputs[0], "input_projection"));
        for (int block = 0; block < 4; ++block) {
            hidden = run_block(hidden, block);
        }

        mx::array pooled = mx::sum(hidden, 1) / mx::array(static_cast<float>(kCellCount));
        mx::array global_hidden = relu(linear_layer(inputs[1], "global_encoder"));
        mx::array combined = mx::concatenate({pooled, global_hidden}, 1);

        mx::array claim_logits = mx::squeeze(conv1x1(hidden, "claim_policy_scorer"), 2);
        mx::array finish_hidden = relu(linear_layer(combined, "finish_policy_scorer.0"));
        mx::array finish_logit = linear_layer(finish_hidden, "finish_policy_scorer.2");
        mx::array policy_logits = mx::concatenate({claim_logits, finish_logit}, 1);

        if (dual_value_) {
            mx::array win_hidden = relu(linear_layer(combined, "win_value_head.0"));
            mx::array win_value = mx::squeeze(
                mx::tanh(linear_layer(win_hidden, "win_value_head.2")),
                1);
            mx::array margin_hidden = relu(linear_layer(combined, "margin_value_head.0"));
            mx::array margin_value = mx::squeeze(
                mx::tanh(linear_layer(margin_hidden, "margin_value_head.2")),
                1);
            return {policy_logits, win_value, margin_value};
        }
        mx::array value_hidden = relu(linear_layer(combined, "value_head.0"));
        mx::array value = mx::squeeze(mx::tanh(linear_layer(value_hidden, "value_head.2")), 1);
        return {policy_logits, value};
    }

    mx::array run_block(const mx::array& hidden, int block) const {
        std::string prefix = "blocks." + std::to_string(block);
        mx::array hidden_transposed = mx::transpose(hidden, {0, 2, 1});
        // Keep channels ordered as dir0, dir1, ..., dir5 to match neighbor_linear.
        mx::array flat_hidden = mx::reshape(
            hidden_transposed,
            {hidden.shape()[0] * hidden.shape()[2], kCellCount});
        mx::array wide_neighbors = mx::matmul(flat_hidden, direction_wide_matrix_);
        mx::array neighbor_stack = mx::reshape(
            mx::transpose(
                mx::reshape(
                    wide_neighbors,
                    {hidden.shape()[0], hidden.shape()[2], 6, kCellCount}),
                {0, 3, 2, 1}),
            {hidden.shape()[0], kCellCount, 6 * hidden.shape()[2]});
        mx::array update = relu(
            conv1x1(hidden, prefix + ".self_linear")
            + conv1x1(neighbor_stack, prefix + ".neighbor_linear"));
        return hidden + update;
    }

    mx::array conv1x1(const mx::array& x, const std::string& prefix) const {
        mx::array weight = require_weight(weights_, prefix + ".weight");
        weight = mx::reshape(weight, {weight.shape()[0], weight.shape()[1]});
        return linear(x, weight, require_weight(weights_, prefix + ".bias"));
    }

    mx::array linear_layer(const mx::array& x, const std::string& prefix) const {
        return linear(
            x,
            require_weight(weights_, prefix + ".weight"),
            require_weight(weights_, prefix + ".bias"));
    }

    std::unordered_map<std::string, mx::array> weights_;
    int batch_size_;
    bool dual_value_;
    mx::array direction_wide_matrix_;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_forward_;
};

}  // namespace

std::unique_ptr<NeuralBatchModel> make_mlx_neural_batch_model(
    const std::string& weights_path,
    int batch_size) {
    return std::make_unique<MlxDirectionalModel>(weights_path, batch_size);
}
