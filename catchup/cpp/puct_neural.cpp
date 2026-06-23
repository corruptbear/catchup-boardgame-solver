#include "puct_neural.hpp"

#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

constexpr int kCellFeatureCount = 4;
constexpr int kFeatureCount = kCellCount * kCellFeatureCount + kMaxActions + 5;
constexpr int kLegalMaskOffset = kCellCount * kCellFeatureCount;
constexpr int kScalarOffset = kLegalMaskOffset + kMaxActions;
std::mutex loader_mutex;

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

    features[kScalarOffset] = static_cast<float>(state.current_player);
    features[kScalarOffset + 1] = static_cast<float>(state.selected.size()) / 3.0F;
    features[kScalarOffset + 2] = static_cast<float>(state.max_claims) / 3.0F;
    features[kScalarOffset + 3] =
        static_cast<float>(state.turn_start_largest) / static_cast<float>(kCellCount);
    features[kScalarOffset + 4] = state.opening_turn ? 1.0F : 0.0F;
}

std::vector<NeuralEvaluation> run_model_batch(
    torch::inductor::AOTIModelPackageLoader& loader,
    const std::vector<TrackedState>& states,
    int input_rows) {
    if (states.empty()) {
        return {};
    }
    if (input_rows < static_cast<int>(states.size())) {
        throw std::runtime_error("neural batch has more states than input rows");
    }

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

    auto input = at::from_blob(
                     features.data(),
                     {input_rows, kFeatureCount},
                     at::TensorOptions().dtype(at::kFloat))
                     .clone()
                     .to(at::kMPS);

    std::vector<at::Tensor> outputs = loader.run({input});
    if (outputs.size() != 2) {
        throw std::runtime_error("neural model must return policy logits and value");
    }

    auto policy = outputs[0].to(at::kCPU).contiguous();
    auto value_tensor = outputs[1].to(at::kCPU).contiguous();
    if (policy.size(0) < input_rows || value_tensor.size(0) < input_rows) {
        throw std::runtime_error("neural model returned fewer rows than the configured batch size");
    }

    const float* all_logits = policy.data_ptr<float>();
    const float* values = value_tensor.data_ptr<float>();
    std::vector<NeuralEvaluation> evaluations;
    evaluations.reserve(states.size());
    for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
        if (legal_actions[state_index].empty()) {
            evaluations.push_back({
                {},
                static_cast<double>(states[state_index].result_for(states[state_index].current_player)),
                states[state_index].current_player,
            });
            continue;
        }

        const float* logits = all_logits + state_index * kMaxActions;
        double max_logit = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < legal_actions[state_index].size(); ++index) {
            max_logit = std::max(
                max_logit,
                static_cast<double>(logits[legal_actions[state_index][index]]));
        }

        std::array<double, kMaxActions> priors{};
        double total = 0.0;
        for (std::size_t index = 0; index < legal_actions[state_index].size(); ++index) {
            int action = legal_actions[state_index][index];
            double weight = std::exp(static_cast<double>(logits[action]) - max_logit);
            priors[action] = weight;
            total += weight;
        }
        for (std::size_t index = 0; index < legal_actions[state_index].size(); ++index) {
            priors[legal_actions[state_index][index]] /= total;
        }

        evaluations.push_back({
            priors,
            static_cast<double>(values[state_index]),
            states[state_index].current_player,
        });
    }
    return evaluations;
}

}  // namespace

struct NeuralEvaluator::Impl {
    explicit Impl(const std::string& package_path)
        : loader(make_loader(package_path)) {}

    torch::inductor::AOTIModelPackageLoader loader;
};

NeuralEvaluator::NeuralEvaluator(const std::string& package_path)
    : impl_(std::make_unique<Impl>(package_path)) {}

NeuralEvaluator::~NeuralEvaluator() = default;

NeuralEvaluation NeuralEvaluator::evaluate(const TrackedState& state) {
    return run_model_batch(impl_->loader, {state}, 1).front();
}

struct BatchedNeuralEvaluator::Impl {
    struct Request {
        explicit Request(const TrackedState& request_state)
            : state(request_state) {}

        TrackedState state;
        std::promise<NeuralEvaluation> result;
    };

    Impl(const std::string& package_path, int requested_batch_size, double requested_wait_ms)
        : loader(make_loader(package_path)),
          batch_size(requested_batch_size),
          wait_ms(requested_wait_ms),
          worker(&Impl::run, this) {
        if (batch_size <= 0) {
            throw std::runtime_error("neural batch size must be positive");
        }
        if (wait_ms < 0.0) {
            throw std::runtime_error("neural batch wait must be non-negative");
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv.notify_all();
        worker.join();
    }

    NeuralEvaluation evaluate(const TrackedState& state) {
        auto request = std::make_shared<Request>(state);
        auto future = request->result.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push_back(request);
        }
        cv.notify_one();
        return future.get();
    }

    void run() {
        while (true) {
            std::vector<std::shared_ptr<Request>> batch = take_batch();
            if (batch.empty()) {
                return;
            }

            try {
                std::vector<TrackedState> states;
                states.reserve(batch.size());
                for (const auto& request : batch) {
                    states.push_back(request->state);
                }

                auto evaluations = run_model_batch(loader, states, batch_size);
                for (std::size_t index = 0; index < batch.size(); ++index) {
                    batch[index]->result.set_value(evaluations[index]);
                }
            } catch (...) {
                auto error = std::current_exception();
                for (auto& request : batch) {
                    request->result.set_exception(error);
                }
            }
        }
    }

    std::vector<std::shared_ptr<Request>> take_batch() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() { return stopping || !queue.empty(); });
        if (stopping && queue.empty()) {
            return {};
        }

        if (!stopping && wait_ms > 0.0 && static_cast<int>(queue.size()) < batch_size) {
            auto deadline = std::chrono::steady_clock::now()
                + std::chrono::duration<double, std::milli>(wait_ms);
            cv.wait_until(lock, deadline, [&]() {
                return stopping || static_cast<int>(queue.size()) >= batch_size;
            });
        }

        int count = std::min<int>(batch_size, queue.size());
        std::vector<std::shared_ptr<Request>> batch;
        batch.reserve(count);
        for (int index = 0; index < count; ++index) {
            batch.push_back(queue.front());
            queue.pop_front();
        }
        return batch;
    }

    torch::inductor::AOTIModelPackageLoader loader;
    int batch_size;
    double wait_ms;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Request>> queue;
    bool stopping = false;
    std::thread worker;
};

BatchedNeuralEvaluator::BatchedNeuralEvaluator(
    const std::string& package_path,
    int batch_size,
    double wait_ms)
    : impl_(std::make_unique<Impl>(package_path, batch_size, wait_ms)) {}

BatchedNeuralEvaluator::~BatchedNeuralEvaluator() = default;

NeuralEvaluation BatchedNeuralEvaluator::evaluate(const TrackedState& state) {
    return impl_->evaluate(state);
}

NeuralPuctMcts::NeuralPuctMcts(
    int simulation_count,
    std::uint64_t seed,
    NeuralEvaluatorBase& evaluator)
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
