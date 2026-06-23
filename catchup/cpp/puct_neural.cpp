#include "puct_neural.hpp"

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kCellFeatureCount = 4;
constexpr int kFeatureCount = kCellCount * kCellFeatureCount + kMaxActions + 5;
constexpr int kLegalMaskOffset = kCellCount * kCellFeatureCount;
constexpr int kScalarOffset = kLegalMaskOffset + kMaxActions;
std::mutex loader_mutex;

void write_state_features(const TrackedState& state, std::array<float, kFeatureCount>& features) {
    features.fill(0.0F);
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

    features[kScalarOffset] = static_cast<float>(state.current_player);
    features[kScalarOffset + 1] = static_cast<float>(state.selected.size()) / 3.0F;
    features[kScalarOffset + 2] = static_cast<float>(state.max_claims) / 3.0F;
    features[kScalarOffset + 3] =
        static_cast<float>(state.turn_start_largest) / static_cast<float>(kCellCount);
    features[kScalarOffset + 4] = state.opening_turn ? 1.0F : 0.0F;
}

}  // namespace

struct NeuralEvaluator::Impl {
    explicit Impl(const std::string& package_path)
        : loader(make_loader(package_path)) {}

    torch::inductor::AOTIModelPackageLoader loader;

private:
    static torch::inductor::AOTIModelPackageLoader make_loader(
        const std::string& package_path) {
        std::lock_guard<std::mutex> lock(loader_mutex);
        return torch::inductor::AOTIModelPackageLoader(package_path, "model", false, 1, -1);
    }
};

NeuralEvaluator::NeuralEvaluator(const std::string& package_path)
    : impl_(std::make_unique<Impl>(package_path)) {}

NeuralEvaluator::~NeuralEvaluator() = default;

NeuralEvaluation NeuralEvaluator::evaluate(const TrackedState& state) {
    ActionList legal_actions = state.legal_actions();
    if (legal_actions.empty()) {
        return {
            {},
            static_cast<double>(state.result_for(state.current_player)),
            state.current_player,
        };
    }

    std::array<float, kFeatureCount> features{};
    write_state_features(state, features);
    auto input = at::from_blob(
                     features.data(),
                     {1, kFeatureCount},
                     at::TensorOptions().dtype(at::kFloat))
                     .clone()
                     .to(at::kMPS);

    std::vector<at::Tensor> outputs = impl_->loader.run({input});
    if (outputs.size() != 2) {
        throw std::runtime_error("neural model must return policy logits and value");
    }

    auto policy = outputs[0].to(at::kCPU).contiguous();
    auto value_tensor = outputs[1].to(at::kCPU).contiguous();
    const float* logits = policy.data_ptr<float>();

    double max_logit = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        max_logit = std::max(max_logit, static_cast<double>(logits[legal_actions[index]]));
    }

    std::array<double, kMaxActions> priors{};
    double total = 0.0;
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        int action = legal_actions[index];
        double weight = std::exp(static_cast<double>(logits[action]) - max_logit);
        priors[action] = weight;
        total += weight;
    }
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        priors[legal_actions[index]] /= total;
    }

    return {
        priors,
        static_cast<double>(value_tensor[0].item<float>()),
        state.current_player,
    };
}

NeuralPuctMcts::NeuralPuctMcts(
    int simulation_count,
    std::uint64_t seed,
    NeuralEvaluator& evaluator)
    : simulations_(simulation_count),
      rng_(seed),
      evaluator_(evaluator) {
    if (simulations_ <= 0) {
        throw std::runtime_error("simulations must be positive");
    }
}

PuctNode* NeuralPuctMcts::search(const TrackedState& root_state) {
    if (root_state.is_terminal()) {
        throw std::runtime_error("cannot search from terminal state");
    }

    nodes_.clear();
    nodes_.push_back(std::make_unique<PuctNode>(root_state));
    PuctNode* root = nodes_.back().get();
    initialize_edges(root, evaluator_.evaluate(root->state).priors);

    for (int simulation = 0; simulation < simulations_; ++simulation) {
        LeafEvaluation evaluation = select_and_evaluate(root);
        backpropagate(evaluation.path, evaluation.value, evaluation.player);
    }
    return root;
}

NeuralPuctMcts::LeafEvaluation NeuralPuctMcts::select_and_evaluate(PuctNode* root) {
    std::vector<PuctNode*> path = {root};
    PuctNode* node = root;

    while (true) {
        if (node->state.is_terminal()) {
            return {
                path,
                static_cast<double>(node->state.result_for(node->state.current_player)),
                node->state.current_player,
            };
        }

        PuctChild* edge = select_child(node);
        if (edge->child == nullptr) {
            PuctNode* child = materialize_child(node, *edge);
            path.push_back(child);
            if (child->state.is_terminal()) {
                return {
                    path,
                    static_cast<double>(child->state.result_for(child->state.current_player)),
                    child->state.current_player,
                };
            }
            NeuralEvaluation evaluation = evaluator_.evaluate(child->state);
            initialize_edges(child, evaluation.priors);
            return {path, evaluation.value, evaluation.player};
        }

        node = edge->child;
        path.push_back(node);
    }
}

PuctChild* NeuralPuctMcts::select_child(PuctNode* node) {
    const double parent_sqrt = std::sqrt(std::max(node->visits, 1));
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<PuctChild*> best_children;

    for (auto& edge : node->action_edges) {
        PuctNode* child = edge.child;
        int child_visits = child == nullptr ? 0 : child->visits;
        double exploitation =
            child == nullptr ? 0.0 : mean_value_for_player(child, node->state.current_player);
        double exploration = 1.4 * edge.prior * parent_sqrt / (1.0 + child_visits);
        double score = exploitation + exploration;

        if (score > best_score) {
            best_score = score;
            best_children = {&edge};
        } else if (score == best_score) {
            best_children.push_back(&edge);
        }
    }

    std::uniform_int_distribution<std::size_t> distribution(0, best_children.size() - 1);
    return best_children[distribution(rng_)];
}

PuctNode* NeuralPuctMcts::materialize_child(PuctNode* node, PuctChild& edge) {
    TrackedState child_state = node->state;
    child_state.apply_action(edge.action);
    nodes_.push_back(std::make_unique<PuctNode>(std::move(child_state), node, edge.action));
    PuctNode* child = nodes_.back().get();
    edge.child = child;
    node->children.push_back(edge);
    return child;
}

void NeuralPuctMcts::initialize_edges(
    PuctNode* node,
    const std::array<double, kMaxActions>& priors) {
    ActionList legal_actions = node->state.legal_actions();
    node->action_edges.reserve(legal_actions.size());
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        int action = legal_actions[index];
        node->action_edges.push_back({
            static_cast<Action>(action),
            nullptr,
            priors[action],
        });
    }
}

void NeuralPuctMcts::backpropagate(
    const std::vector<PuctNode*>& path,
    double value,
    int value_player) {
    for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
        PuctNode* node = *iterator;
        ++node->visits;
        node->total_value += node->state.current_player == value_player ? value : -value;
    }
}

double NeuralPuctMcts::mean_value_for_player(const PuctNode* node, int player) {
    double value = node->mean_value();
    if (node->state.current_player == player) {
        return value;
    }
    return -value;
}
