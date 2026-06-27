#pragma once

#include "puct_mcts.hpp"
#include "tracked_state.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

enum class NeuralDevice {
    Cpu,
    Mps,
};

enum class NeuralBackend {
    Aoti,
    Mlx,
};

enum class NeuralValueTarget {
    WinLoss,
    TanhMarginScale6,
};

NeuralDevice parse_neural_device(const std::string& text);
const char* neural_device_label(NeuralDevice device);
NeuralBackend parse_neural_backend(const std::string& text);
const char* neural_backend_label(NeuralBackend backend);
NeuralValueTarget parse_neural_value_target(const std::string& text);
const char* neural_value_target_label(NeuralValueTarget target);
NeuralValueTarget infer_neural_value_target_from_model_path(const std::string& model_path);

struct NeuralEvaluation {
    std::array<double, kMaxActions> priors{};
    double value = 0.0;
    int player = kPlayerOne;
};

struct NeuralBatchStats {
    std::uint64_t requests = 0;
    std::uint64_t batches = 0;
    std::uint64_t batch_items = 0;
    std::uint64_t full_batches = 0;
    std::uint64_t fill_waits = 0;
    std::uint64_t deadline_batches = 0;
    std::uint64_t fill_wait_ns = 0;
    std::uint64_t model_time_ns = 0;
    std::uint64_t feature_time_ns = 0;
    std::uint64_t input_time_ns = 0;
    std::uint64_t inference_time_ns = 0;
    std::uint64_t output_time_ns = 0;
    std::uint64_t postprocess_time_ns = 0;
    std::uint64_t request_latency_ns = 0;
};

struct NeuralBatchTiming {
    std::uint64_t feature_ns = 0;
    std::uint64_t input_ns = 0;
    std::uint64_t inference_ns = 0;
    std::uint64_t output_ns = 0;
    std::uint64_t postprocess_ns = 0;
};

struct NeuralPuctConfig {
    double root_noise_epsilon = 0.0;
    double root_dirichlet_total_concentration = 10.0;
    double root_noise_reference_actions = kCellCount;
    double root_noise_action_power = 0.5;
    double root_noise_empty_power = 1.0;
    NeuralValueTarget value_target = NeuralValueTarget::WinLoss;
};

class NeuralEvaluatorBase {
public:
    virtual ~NeuralEvaluatorBase() = default;
    virtual NeuralEvaluation evaluate(const TrackedState& state) = 0;
};

class NeuralBatchModel {
public:
    virtual ~NeuralBatchModel() = default;
    virtual std::vector<NeuralEvaluation> evaluate_batch(
        const std::vector<TrackedState>& states,
        NeuralBatchTiming* timing) = 0;
};

std::vector<NeuralEvaluation> build_neural_evaluations(
    const std::vector<TrackedState>& states,
    const std::vector<ActionList>& legal_actions,
    const float* all_logits,
    const float* values);

std::unique_ptr<NeuralBatchModel> make_aoti_neural_batch_model(
    const std::string& package_path,
    int batch_size,
    NeuralDevice device);

std::unique_ptr<NeuralBatchModel> make_mlx_neural_batch_model(
    const std::string& weights_path,
    int batch_size);

class NeuralEvaluator : public NeuralEvaluatorBase {
public:
    explicit NeuralEvaluator(std::unique_ptr<NeuralBatchModel> model);
    ~NeuralEvaluator();

    NeuralEvaluator(const NeuralEvaluator&) = delete;
    NeuralEvaluator& operator=(const NeuralEvaluator&) = delete;

    NeuralEvaluation evaluate(const TrackedState& state);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class BatchedNeuralEvaluator : public NeuralEvaluatorBase {
public:
    BatchedNeuralEvaluator(
        std::unique_ptr<NeuralBatchModel> model,
        int batch_size,
        double wait_ms);
    ~BatchedNeuralEvaluator();

    BatchedNeuralEvaluator(const BatchedNeuralEvaluator&) = delete;
    BatchedNeuralEvaluator& operator=(const BatchedNeuralEvaluator&) = delete;

    NeuralEvaluation evaluate(const TrackedState& state);
    NeuralBatchStats batch_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NeuralPuctMcts {
public:
    NeuralPuctMcts(
        int simulation_count,
        std::uint64_t seed,
        NeuralEvaluatorBase& evaluator,
        NeuralPuctConfig config = {});

    PuctNode* search(const TrackedState& root_state);

private:
    struct LeafEvaluation {
        std::vector<PuctNode*> path;
        double value = 0.0;
        int player = kPlayerOne;
    };

    LeafEvaluation select_and_evaluate(PuctNode* root);
    PuctChild* select_child(PuctNode* node);
    PuctNode* materialize_child(PuctNode* node, PuctChild& edge);
    void initialize_edges(PuctNode* node, const std::array<double, kMaxActions>& priors);
    void add_root_dirichlet_noise(PuctNode* root);
    static void backpropagate(
        const std::vector<PuctNode*>& path,
        double value,
        int value_player);
    static double mean_value_for_player(const PuctNode* node, int player);

    int simulations_;
    std::mt19937_64 rng_;
    NeuralEvaluatorBase& evaluator_;
    NeuralPuctConfig config_;
    std::vector<std::unique_ptr<PuctNode>> nodes_;
};
