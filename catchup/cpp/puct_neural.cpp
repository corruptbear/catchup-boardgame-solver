#include "puct_neural.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

}  // namespace

NeuralDevice parse_neural_device(const std::string& text) {
    if (text == "cpu") {
        return NeuralDevice::Cpu;
    }
    if (text == "mps") {
        return NeuralDevice::Mps;
    }
    throw std::runtime_error("neural device must be cpu or mps");
}

const char* neural_device_label(NeuralDevice device) {
    return device == NeuralDevice::Cpu ? "cpu" : "mps";
}

NeuralBackend parse_neural_backend(const std::string& text) {
    if (text == "aoti") {
        return NeuralBackend::Aoti;
    }
    if (text == "mlx") {
        return NeuralBackend::Mlx;
    }
    throw std::runtime_error("neural backend must be aoti or mlx");
}

const char* neural_backend_label(NeuralBackend backend) {
    return backend == NeuralBackend::Mlx ? "mlx" : "aoti";
}

std::vector<NeuralEvaluation> build_neural_evaluations(
    const std::vector<TrackedState>& states,
    const std::vector<ActionList>& legal_actions,
    const float* all_logits,
    const float* values) {
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

struct NeuralEvaluator::Impl {
    explicit Impl(std::unique_ptr<NeuralBatchModel> requested_model)
        : model(std::move(requested_model)) {}

    std::unique_ptr<NeuralBatchModel> model;
};

NeuralEvaluator::NeuralEvaluator(std::unique_ptr<NeuralBatchModel> model)
    : impl_(std::make_unique<Impl>(std::move(model))) {}

NeuralEvaluator::~NeuralEvaluator() = default;

NeuralEvaluation NeuralEvaluator::evaluate(const TrackedState& state) {
    return impl_->model->evaluate_batch({state}, nullptr).front();
}

struct BatchedNeuralEvaluator::Impl {
    struct Request {
        explicit Request(const TrackedState& request_state)
            : state(request_state),
              created_at(Clock::now()) {}

        TrackedState state;
        Clock::time_point created_at;
        std::promise<NeuralEvaluation> result;
    };

    Impl(
        std::unique_ptr<NeuralBatchModel> requested_model,
        int requested_batch_size,
        double requested_wait_ms)
        : model(std::move(requested_model)),
          batch_size(requested_batch_size),
          wait_ms(requested_wait_ms) {
        if (batch_size <= 0) {
            throw std::runtime_error("neural batch size must be positive");
        }
        if (wait_ms < 0.0) {
            throw std::runtime_error("neural batch wait must be non-negative");
        }
        worker = std::thread(&Impl::run, this);
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
        request_count.fetch_add(1, std::memory_order_relaxed);
        cv.notify_one();
        return future.get();
    }

    NeuralBatchStats batch_stats() const {
        return {
            request_count.load(std::memory_order_relaxed),
            batch_count.load(std::memory_order_relaxed),
            batch_item_count.load(std::memory_order_relaxed),
            full_batch_count.load(std::memory_order_relaxed),
            fill_wait_count.load(std::memory_order_relaxed),
            deadline_batch_count.load(std::memory_order_relaxed),
            fill_wait_nanoseconds.load(std::memory_order_relaxed),
            model_time_nanoseconds.load(std::memory_order_relaxed),
            feature_time_nanoseconds.load(std::memory_order_relaxed),
            input_time_nanoseconds.load(std::memory_order_relaxed),
            inference_time_nanoseconds.load(std::memory_order_relaxed),
            output_time_nanoseconds.load(std::memory_order_relaxed),
            postprocess_time_nanoseconds.load(std::memory_order_relaxed),
            request_latency_nanoseconds.load(std::memory_order_relaxed),
        };
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

                auto model_started = Clock::now();
                NeuralBatchTiming timing;
                auto evaluations = model->evaluate_batch(states, &timing);
                auto completed_at = Clock::now();
                model_time_nanoseconds.fetch_add(
                    elapsed_ns(model_started, completed_at),
                    std::memory_order_relaxed);
                feature_time_nanoseconds.fetch_add(timing.feature_ns, std::memory_order_relaxed);
                input_time_nanoseconds.fetch_add(timing.input_ns, std::memory_order_relaxed);
                inference_time_nanoseconds.fetch_add(timing.inference_ns, std::memory_order_relaxed);
                output_time_nanoseconds.fetch_add(timing.output_ns, std::memory_order_relaxed);
                postprocess_time_nanoseconds.fetch_add(
                    timing.postprocess_ns,
                    std::memory_order_relaxed);
                for (std::size_t index = 0; index < batch.size(); ++index) {
                    request_latency_nanoseconds.fetch_add(
                        elapsed_ns(batch[index]->created_at, completed_at),
                        std::memory_order_relaxed);
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

        bool waited_to_fill = false;
        bool deadline_used = false;
        auto wait_started = Clock::now();
        if (!stopping && wait_ms > 0.0 && static_cast<int>(queue.size()) < batch_size) {
            waited_to_fill = true;
            auto deadline = std::chrono::steady_clock::now()
                + std::chrono::duration<double, std::milli>(wait_ms);
            cv.wait_until(lock, deadline, [&]() {
                return stopping || static_cast<int>(queue.size()) >= batch_size;
            });
            auto wait_finished = Clock::now();
            fill_wait_nanoseconds.fetch_add(
                elapsed_ns(wait_started, wait_finished),
                std::memory_order_relaxed);
            fill_wait_count.fetch_add(1, std::memory_order_relaxed);
            deadline_used = !stopping && static_cast<int>(queue.size()) < batch_size;
        }

        int count = std::min<int>(batch_size, queue.size());
        batch_count.fetch_add(1, std::memory_order_relaxed);
        batch_item_count.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
        if (count == batch_size) {
            full_batch_count.fetch_add(1, std::memory_order_relaxed);
        }
        if (waited_to_fill && deadline_used) {
            deadline_batch_count.fetch_add(1, std::memory_order_relaxed);
        }
        std::vector<std::shared_ptr<Request>> batch;
        batch.reserve(count);
        for (int index = 0; index < count; ++index) {
            batch.push_back(queue.front());
            queue.pop_front();
        }
        return batch;
    }

    std::unique_ptr<NeuralBatchModel> model;
    int batch_size;
    double wait_ms;
    std::atomic<std::uint64_t> request_count{0};
    std::atomic<std::uint64_t> batch_count{0};
    std::atomic<std::uint64_t> batch_item_count{0};
    std::atomic<std::uint64_t> full_batch_count{0};
    std::atomic<std::uint64_t> fill_wait_count{0};
    std::atomic<std::uint64_t> deadline_batch_count{0};
    std::atomic<std::uint64_t> fill_wait_nanoseconds{0};
    std::atomic<std::uint64_t> model_time_nanoseconds{0};
    std::atomic<std::uint64_t> feature_time_nanoseconds{0};
    std::atomic<std::uint64_t> input_time_nanoseconds{0};
    std::atomic<std::uint64_t> inference_time_nanoseconds{0};
    std::atomic<std::uint64_t> output_time_nanoseconds{0};
    std::atomic<std::uint64_t> postprocess_time_nanoseconds{0};
    std::atomic<std::uint64_t> request_latency_nanoseconds{0};
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Request>> queue;
    bool stopping = false;
    std::thread worker;
};

BatchedNeuralEvaluator::BatchedNeuralEvaluator(
    std::unique_ptr<NeuralBatchModel> model,
    int batch_size,
    double wait_ms)
    : impl_(std::make_unique<Impl>(std::move(model), batch_size, wait_ms)) {}

BatchedNeuralEvaluator::~BatchedNeuralEvaluator() = default;

NeuralEvaluation BatchedNeuralEvaluator::evaluate(const TrackedState& state) {
    return impl_->evaluate(state);
}

NeuralBatchStats BatchedNeuralEvaluator::batch_stats() const {
    return impl_->batch_stats();
}

NeuralPuctMcts::NeuralPuctMcts(
    int simulation_count,
    std::uint64_t seed,
    NeuralEvaluatorBase& evaluator,
    NeuralPuctConfig config)
    : simulations_(simulation_count),
      rng_(seed),
      evaluator_(evaluator),
      config_(config) {
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
    add_root_dirichlet_noise(root);

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

void NeuralPuctMcts::add_root_dirichlet_noise(PuctNode* root) {
    if (config_.root_noise_epsilon <= 0.0) {
        return;
    }

    const double legal_count = static_cast<double>(root->action_edges.size());
    const double per_action_alpha = config_.root_dirichlet_total_concentration / legal_count;
    const double action_ratio = legal_count / config_.root_noise_reference_actions;
    const double empty_ratio =
        static_cast<double>(root->state.empty_count()) / static_cast<double>(kCellCount);
    const double effective_epsilon =
        config_.root_noise_epsilon
        * std::pow(action_ratio, config_.root_noise_action_power)
        * std::pow(empty_ratio, config_.root_noise_empty_power);

    std::gamma_distribution<double> distribution(per_action_alpha, 1.0);
    std::vector<double> noise;
    noise.reserve(root->action_edges.size());

    double total = 0.0;
    for (std::size_t index = 0; index < root->action_edges.size(); ++index) {
        double value = distribution(rng_);
        noise.push_back(value);
        total += value;
    }

    const double model_weight = 1.0 - effective_epsilon;
    for (std::size_t index = 0; index < root->action_edges.size(); ++index) {
        root->action_edges[index].prior =
            model_weight * root->action_edges[index].prior
            + effective_epsilon * noise[index] / total;
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
