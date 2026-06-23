#include "puct_mcts.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

int find_root_const(const TrackedState& state, int cell) {
    int root = cell;
    while (state.parents[root] != root) {
        root = state.parents[root];
    }
    return root;
}

bool append_unique_root(std::array<int, 6>& roots, int& count, int root) {
    for (int index = 0; index < count; ++index) {
        if (roots[index] == root) {
            return false;
        }
    }
    roots[count++] = root;
    return true;
}

struct ClaimPriorFeatures {
    double score = 1.0;
    bool breaks_global_largest_record = false;
};

ClaimPriorFeatures score_claim_action(const TrackedState& state, int cell) {
    int player = state.current_player;
    int opponent = other_player(player);
    std::array<int, 6> own_roots{};
    std::array<int, 6> opponent_roots{};
    int adjacent_own_components_count = 0;
    int adjacent_opponent_components_count = 0;

    for (int neighbor : board().neighbors[cell]) {
        int owner = state.owners[neighbor];
        if (owner == player) {
            append_unique_root(
                own_roots,
                adjacent_own_components_count,
                find_root_const(state, neighbor));
        } else if (owner == opponent) {
            append_unique_root(
                opponent_roots,
                adjacent_opponent_components_count,
                find_root_const(state, neighbor));
        }
    }

    int merged_size = 1;
    for (int index = 0; index < adjacent_own_components_count; ++index) {
        merged_size += state.sizes[own_roots[index]];
    }

    bool record_already_broken =
        !state.opening_turn && state.largest_group_size() > state.turn_start_largest;
    bool breaks_global_largest_record =
        !state.opening_turn && !record_already_broken && merged_size > state.turn_start_largest;

    double score = 1.0;
    if (adjacent_own_components_count == 0) {
        score += 0.8;
    } else {
        score += 0.15 * merged_size;
        score += 0.35 * adjacent_own_components_count;
    }

    if (adjacent_own_components_count >= 2) {
        score += 2.5 + 0.20 * merged_size;
    }

    for (int index = 0; index < adjacent_opponent_components_count; ++index) {
        score += 0.35 + 0.04 * state.sizes[opponent_roots[index]];
    }

    if (breaks_global_largest_record) {
        score *= adjacent_own_components_count >= 2 ? 0.70 : 0.35;
    } else {
        score += 0.45;
    }

    return {std::max(score, 0.05), breaks_global_largest_record};
}

}  // namespace

PuctPriorMode parse_puct_prior_mode(const std::string& text) {
    if (text == "flat") {
        return PuctPriorMode::Flat;
    }
    if (text == "heuristic") {
        return PuctPriorMode::Heuristic;
    }
    throw std::runtime_error("puct prior must be flat or heuristic");
}

PuctRolloutMode parse_puct_rollout_mode(const std::string& text) {
    if (text == "flat") {
        return PuctRolloutMode::Flat;
    }
    if (text == "biased") {
        return PuctRolloutMode::Biased;
    }
    throw std::runtime_error("puct rollout must be flat or biased");
}

std::string puct_prior_mode_name(PuctPriorMode mode) {
    if (mode == PuctPriorMode::Flat) {
        return "flat";
    }
    return "heuristic";
}

std::string puct_rollout_mode_name(PuctRolloutMode mode) {
    if (mode == PuctRolloutMode::Flat) {
        return "flat";
    }
    return "biased";
}

std::array<double, kMaxActions> HeuristicActionPrior::operator()(
    const TrackedState& state) const {
    return (*this)(state, state.legal_actions());
}

std::array<double, kMaxActions> HeuristicActionPrior::operator()(
    const TrackedState& state,
    const ActionList& legal_actions) const {
    std::array<double, kMaxActions> priors{};
    if (legal_actions.empty()) {
        return priors;
    }

    int claim_count = 0;
    int safe_claim_count = 0;
    double claim_score_total = 0.0;
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        int action = legal_actions[index];
        if (action == kFinish) {
            continue;
        }
        ClaimPriorFeatures features = score_claim_action(state, action);
        priors[action] = features.score;
        ++claim_count;
        claim_score_total += features.score;
        if (!features.breaks_global_largest_record) {
            ++safe_claim_count;
        }
    }

    if (!state.selected.empty()) {
        if (claim_count == 0) {
            priors[kFinish] = 1.0;
        } else if (safe_claim_count == 0) {
            priors[kFinish] = std::max(1.2, 0.25 * claim_score_total / claim_count);
        } else {
            priors[kFinish] = 0.15;
        }
    }

    double total = 0.0;
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        total += priors[legal_actions[index]];
    }
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        priors[legal_actions[index]] /= total;
    }
    return priors;
}

PuctNode::PuctNode(TrackedState node_state, PuctNode* parent_node, int action)
    : state(std::move(node_state)),
      parent(parent_node),
      action_from_parent(action) {}

double PuctNode::mean_value() const {
    return visits == 0 ? 0.0 : total_value / static_cast<double>(visits);
}

PuctMcts::PuctMcts(int simulation_count, std::uint64_t seed, PuctConfig config)
    : simulations_(simulation_count), rng_(seed), config_(config) {
    if (simulations_ <= 0) {
        throw std::runtime_error("simulations must be positive");
    }
}

PuctNode* PuctMcts::search(const TrackedState& root_state) {
    nodes_.clear();
    nodes_.push_back(std::make_unique<PuctNode>(root_state));
    PuctNode* root = nodes_.back().get();
    initialize_edges(root);

    for (int simulation = 0; simulation < simulations_; ++simulation) {
        auto path = select_and_expand(root);
        TrackedState terminal =
            config_.rollout == PuctRolloutMode::Flat
                ? flat_random_rollout(path.back()->state)
                : biased_random_rollout(path.back()->state);
        backpropagate(path, terminal);
    }
    return root;
}

std::vector<PuctNode*> PuctMcts::select_and_expand(PuctNode* root) {
    std::vector<PuctNode*> path = {root};
    PuctNode* node = root;

    while (!node->state.is_terminal()) {
        PuctChild* edge = select_child(node);
        if (edge->child == nullptr) {
            PuctNode* child = materialize_child(node, *edge);
            path.push_back(child);
            return path;
        }
        node = edge->child;
        path.push_back(node);
    }

    return path;
}

PuctNode* PuctMcts::materialize_child(PuctNode* node, PuctChild& edge) {
    TrackedState child_state = node->state;
    child_state.apply_action(edge.action);
    nodes_.push_back(std::make_unique<PuctNode>(std::move(child_state), node, edge.action));
    PuctNode* child = nodes_.back().get();
    initialize_edges(child);
    edge.child = child;
    node->children.push_back(edge);
    return child;
}

PuctChild* PuctMcts::select_child(PuctNode* node) {
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

int PuctMcts::sample_flat_rollout_action(const ActionList& actions) {
    std::uniform_int_distribution<std::size_t> distribution(0, actions.size() - 1);
    return actions[distribution(rng_)];
}

int PuctMcts::sample_biased_rollout_action(
    const TrackedState& state,
    const ActionList& actions) {
    auto priors = heuristic_prior_(state, actions);
    std::array<double, kMaxActions> weights{};
    for (std::size_t index = 0; index < actions.size(); ++index) {
        weights[index] = priors[actions[index]];
    }
    std::discrete_distribution<std::size_t> distribution(
        weights.begin(),
        weights.begin() + actions.size());
    return actions[distribution(rng_)];
}

std::array<double, kMaxActions> PuctMcts::action_priors(
    const TrackedState& state,
    const ActionList& legal_actions) const {
    if (config_.prior == PuctPriorMode::Heuristic) {
        return heuristic_prior_(state, legal_actions);
    }

    std::array<double, kMaxActions> priors{};
    if (legal_actions.empty()) {
        return priors;
    }
    double prior = 1.0 / static_cast<double>(legal_actions.size());
    for (std::size_t index = 0; index < legal_actions.size(); ++index) {
        priors[legal_actions[index]] = prior;
    }
    return priors;
}

TrackedState PuctMcts::flat_random_rollout(const TrackedState& state) {
    TrackedState current = state;
    while (!current.is_terminal()) {
        auto actions = current.legal_actions();
        current.apply_action(sample_flat_rollout_action(actions));
    }
    return current;
}

TrackedState PuctMcts::biased_random_rollout(const TrackedState& state) {
    TrackedState current = state;
    while (!current.is_terminal()) {
        auto actions = current.legal_actions();
        current.apply_action(sample_biased_rollout_action(current, actions));
    }
    return current;
}

void PuctMcts::initialize_edges(PuctNode* node) {
    ActionList legal_actions = node->state.legal_actions();
    auto priors = action_priors(node->state, legal_actions);
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

void PuctMcts::backpropagate(const std::vector<PuctNode*>& path, const TrackedState& terminal) {
    for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
        PuctNode* node = *iterator;
        ++node->visits;
        node->total_value += terminal.result_for(node->state.current_player);
    }
}

double PuctMcts::mean_value_for_player(const PuctNode* node, int player) {
    double value = node->mean_value();
    if (node->state.current_player == player) {
        return value;
    }
    return -value;
}
