#pragma once

#include "tracked_state.hpp"

#include <array>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

enum class PuctPriorMode {
    Flat,
    Heuristic,
};

enum class PuctRolloutMode {
    Flat,
    Biased,
};

struct PuctConfig {
    PuctPriorMode prior = PuctPriorMode::Heuristic;
    PuctRolloutMode rollout = PuctRolloutMode::Biased;
};

PuctPriorMode parse_puct_prior_mode(const std::string& text);
PuctRolloutMode parse_puct_rollout_mode(const std::string& text);
std::string puct_prior_mode_name(PuctPriorMode mode);
std::string puct_rollout_mode_name(PuctRolloutMode mode);

struct HeuristicActionPrior {
    std::array<double, kMaxActions> operator()(const TrackedState& state) const;
    std::array<double, kMaxActions> operator()(
        const TrackedState& state,
        const ActionList& legal_actions) const;
};

struct PuctChild {
    Action action = 0;
    struct PuctNode* child = nullptr;
    double prior = 0.0;
};

struct PuctNode {
    explicit PuctNode(TrackedState node_state, PuctNode* parent_node = nullptr, int action = -1);

    TrackedState state;
    PuctNode* parent = nullptr;
    std::int8_t action_from_parent = -1;
    std::vector<PuctChild> children;
    std::vector<PuctChild> action_edges;
    int visits = 0;
    double total_value = 0.0;

    double mean_value() const;
};

class PuctMcts {
public:
    PuctMcts(int simulation_count, std::uint64_t seed, PuctConfig config = {});

    PuctNode* search(const TrackedState& root_state);

private:
    std::vector<PuctNode*> select_and_expand(PuctNode* root);
    PuctChild* select_child(PuctNode* node);
    PuctNode* materialize_child(PuctNode* node, PuctChild& edge);
    int sample_flat_rollout_action(const ActionList& actions);
    int sample_biased_rollout_action(const TrackedState& state, const ActionList& actions);
    std::array<double, kMaxActions> action_priors(
        const TrackedState& state,
        const ActionList& legal_actions) const;
    TrackedState flat_random_rollout(const TrackedState& state);
    TrackedState biased_random_rollout(const TrackedState& state);
    void initialize_edges(PuctNode* node);
    static void backpropagate(const std::vector<PuctNode*>& path, const TrackedState& terminal);
    static double mean_value_for_player(const PuctNode* node, int player);

    int simulations_;
    std::mt19937_64 rng_;
    PuctConfig config_;
    HeuristicActionPrior heuristic_prior_;
    std::vector<std::unique_ptr<PuctNode>> nodes_;
};
