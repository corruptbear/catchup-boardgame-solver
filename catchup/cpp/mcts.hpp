#pragma once

#include "tracked_state.hpp"

#include <memory>
#include <random>
#include <utility>
#include <vector>

struct Node {
    explicit Node(TrackedState node_state, Node* parent_node = nullptr, int action = -1);

    TrackedState state;
    Node* parent = nullptr;
    std::int8_t action_from_parent = -1;
    std::vector<std::pair<Action, Node*>> children;
    int visits = 0;
    double total_value = 0.0;
    ActionList untried_actions;

    double mean_value() const;
};

class Mcts {
public:
    Mcts(int simulation_count, std::uint64_t seed);

    Node* search(const TrackedState& root_state);

private:
    std::vector<Node*> select_and_expand(Node* root);
    Node* select_child(Node* node);
    int pop_random_untried_action(Node* node);
    TrackedState rollout(const TrackedState& state);
    static void backpropagate(const std::vector<Node*>& path, const TrackedState& terminal);
    static double mean_value_for_player(const Node* node, int player);

    int simulations_;
    std::mt19937_64 rng_;
    std::vector<std::unique_ptr<Node>> nodes_;
};
