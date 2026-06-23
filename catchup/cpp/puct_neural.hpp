#pragma once

#include "puct_mcts.hpp"
#include "tracked_state.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

struct NeuralEvaluation {
    std::array<double, kMaxActions> priors{};
    double value = 0.0;
    int player = kPlayerOne;
};

class NeuralEvaluatorBase {
public:
    virtual ~NeuralEvaluatorBase() = default;
    virtual NeuralEvaluation evaluate(const TrackedState& state) = 0;
};

class NeuralEvaluator : public NeuralEvaluatorBase {
public:
    explicit NeuralEvaluator(const std::string& package_path);
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
        const std::string& package_path,
        int batch_size,
        double wait_ms);
    ~BatchedNeuralEvaluator();

    BatchedNeuralEvaluator(const BatchedNeuralEvaluator&) = delete;
    BatchedNeuralEvaluator& operator=(const BatchedNeuralEvaluator&) = delete;

    NeuralEvaluation evaluate(const TrackedState& state);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class NeuralPuctMcts {
public:
    NeuralPuctMcts(int simulation_count, std::uint64_t seed, NeuralEvaluatorBase& evaluator);

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
    static void backpropagate(
        const std::vector<PuctNode*>& path,
        double value,
        int value_player);
    static double mean_value_for_player(const PuctNode* node, int player);

    int simulations_;
    std::mt19937_64 rng_;
    NeuralEvaluatorBase& evaluator_;
    std::vector<std::unique_ptr<PuctNode>> nodes_;
};
