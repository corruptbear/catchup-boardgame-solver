#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

Node::Node(TrackedState node_state, Node* parent_node, int action)
    : state(std::move(node_state)),
      parent(parent_node),
      action_from_parent(action),
      untried_actions(state.legal_actions()) {}

double Node::mean_value() const {
    return visits == 0 ? 0.0 : total_value / static_cast<double>(visits);
}

Mcts::Mcts(int simulation_count, std::uint64_t seed)
    : simulations_(simulation_count), rng_(seed) {
    if (simulations_ <= 0) {
        throw std::runtime_error("simulations must be positive");
    }
}

Node* Mcts::search(const TrackedState& root_state) {
    nodes_.clear();
    nodes_.push_back(std::make_unique<Node>(root_state));
    Node* root = nodes_.back().get();

    for (int simulation = 0; simulation < simulations_; ++simulation) {
        auto path = select_and_expand(root);
        TrackedState terminal = rollout(path.back()->state);
        backpropagate(path, terminal);
    }
    return root;
}

std::vector<Node*> Mcts::select_and_expand(Node* root) {
    std::vector<Node*> path = {root};
    Node* node = root;

    while (!node->state.is_terminal() && node->untried_actions.empty() && !node->children.empty()) {
        node = select_child(node);
        path.push_back(node);
    }

    if (node->state.is_terminal()) {
        return path;
    }

    int action = pop_random_untried_action(node);
    TrackedState child_state = node->state;
    child_state.apply_action(action);
    nodes_.push_back(std::make_unique<Node>(std::move(child_state), node, action));
    Node* child = nodes_.back().get();
    node->children.push_back({static_cast<Action>(action), child});
    path.push_back(child);
    return path;
}

Node* Mcts::select_child(Node* node) {
    const int parent_visits = std::max(node->visits, 1);
    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<Node*> best_children;

    for (auto& [action, child] : node->children) {
        (void) action;
        double score = 0.0;
        if (child->visits == 0) {
            score = std::numeric_limits<double>::infinity();
        } else {
            double exploitation = mean_value_for_player(child, node->state.current_player);
            double exploration = 1.4 * std::sqrt(std::log(parent_visits) / child->visits);
            score = exploitation + exploration;
        }

        if (score > best_score) {
            best_score = score;
            best_children = {child};
        } else if (score == best_score) {
            best_children.push_back(child);
        }
    }

    std::uniform_int_distribution<std::size_t> distribution(0, best_children.size() - 1);
    return best_children[distribution(rng_)];
}

int Mcts::pop_random_untried_action(Node* node) {
    std::uniform_int_distribution<std::size_t> distribution(0, node->untried_actions.size() - 1);
    std::size_t index = distribution(rng_);
    int action = node->untried_actions[index];
    node->untried_actions.erase_unordered(index);
    return action;
}

TrackedState Mcts::rollout(const TrackedState& state) {
    TrackedState current = state;
    while (!current.is_terminal()) {
        auto actions = current.legal_actions();
        std::uniform_int_distribution<std::size_t> distribution(0, actions.size() - 1);
        current.apply_action(actions[distribution(rng_)]);
    }
    return current;
}

void Mcts::backpropagate(const std::vector<Node*>& path, const TrackedState& terminal) {
    for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
        Node* node = *iterator;
        ++node->visits;
        node->total_value += terminal.result_for(node->state.current_player);
    }
}

double Mcts::mean_value_for_player(const Node* node, int player) {
    double value = node->mean_value();
    if (node->state.current_player == player) {
        return value;
    }
    return -value;
}
