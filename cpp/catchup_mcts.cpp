#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kEmpty = -1;
constexpr int kPlayerOne = 0;
constexpr int kPlayerTwo = 1;
constexpr int kCellCount = 61;
constexpr int kFinish = kCellCount;
constexpr int kEarlyWinCheckMinFilledCells = 30;

struct Board {
    std::array<std::vector<int>, kCellCount> neighbors;

    Board() {
        std::vector<std::pair<int, int>> coords;
        coords.reserve(kCellCount);
        std::unordered_map<std::string, int> index_by_coord;
        constexpr int radius = 4;

        for (int r = -radius; r <= radius; ++r) {
            int q_min = std::max(-radius, -r - radius);
            int q_max = std::min(radius, -r + radius);
            for (int q = q_min; q <= q_max; ++q) {
                int index = static_cast<int>(coords.size());
                coords.push_back({q, r});
                index_by_coord[key(q, r)] = index;
            }
        }

        const std::array<std::pair<int, int>, 6> directions = {{
            {1, 0},
            {-1, 0},
            {0, 1},
            {0, -1},
            {1, -1},
            {-1, 1},
        }};
        for (int index = 0; index < kCellCount; ++index) {
            auto [q, r] = coords[index];
            for (auto [dq, dr] : directions) {
                auto found = index_by_coord.find(key(q + dq, r + dr));
                if (found != index_by_coord.end()) {
                    neighbors[index].push_back(found->second);
                }
            }
        }
    }

    static std::string key(int q, int r) {
        return std::to_string(q) + "," + std::to_string(r);
    }
};

const Board& board() {
    static const Board singleton;
    return singleton;
}

int other_player(int player) {
    return player == kPlayerOne ? kPlayerTwo : kPlayerOne;
}

int compare_size_vectors(const std::vector<int>& first, const std::vector<int>& second) {
    const std::size_t max_size = std::max(first.size(), second.size());
    for (std::size_t index = 0; index < max_size; ++index) {
        int first_size = index < first.size() ? first[index] : 0;
        int second_size = index < second.size() ? second[index] : 0;
        if (first_size != second_size) {
            return first_size > second_size ? 1 : -1;
        }
    }
    return 0;
}

std::uint64_t cell_bit(int cell) {
    return std::uint64_t{1} << cell;
}

int pop_first_cell(std::uint64_t& mask) {
    int cell = __builtin_ctzll(mask);
    mask &= mask - 1;
    return cell;
}

int first_cell(std::uint64_t mask) {
    return __builtin_ctzll(mask);
}

int popcount(std::uint64_t mask) {
    return __builtin_popcountll(mask);
}

struct EmptyFloodResult {
    std::uint64_t cells = 0;
    std::array<std::uint64_t, 2> adjacency{};
};

struct TrackedState {
    std::array<int, kCellCount> owners{};
    std::array<int, kCellCount> parents{};
    std::array<int, kCellCount> sizes{};
    std::array<std::uint64_t, 2> roots_mask{};
    std::array<std::array<int, kCellCount + 1>, 2> size_histogram{};
    std::array<int, 2> max_group_size{};
    int empty_count_cached = kCellCount;
    std::array<int, kCellCount> empty_component_of{};
    std::array<std::uint64_t, kCellCount> empty_component_cells{};
    std::uint64_t empty_roots_mask = 0;
    std::array<std::array<std::uint64_t, 2>, kCellCount> empty_adjacency{};
    std::array<std::array<std::uint64_t, kCellCount>, 2> claimed_adjacent_empty{};
    int current_player = kPlayerOne;
    std::vector<int> selected;
    int max_claims = 1;
    int turn_start_largest = 0;
    bool opening_turn = true;
    int completed_turns = 0;

    TrackedState() {
        owners.fill(kEmpty);
        rebuild_tracking();
    }

    void rebuild_from_owners() {
        rebuild_tracking();
    }

    int empty_count() const {
        return empty_count_cached;
    }

    std::vector<int> group_sizes(int player) const {
        std::vector<int> result;
        for (int size = max_group_size[player]; size > 0; --size) {
            int count = size_histogram[player][size];
            for (int index = 0; index < count; ++index) {
                result.push_back(size);
            }
        }
        return result;
    }

    int largest_group_size() const {
        return std::max(max_group_size[kPlayerOne], max_group_size[kPlayerTwo]);
    }

    std::vector<int> reachable_group_bounds(int player) const {
        std::vector<int> bounds;
        std::uint64_t visited_claimed = 0;
        std::uint64_t visited_empty = 0;

        while (true) {
            std::vector<int> stack;
            std::uint64_t unvisited_claimed = roots_mask[player] & ~visited_claimed;
            std::uint64_t unvisited_empty = empty_roots_mask & ~visited_empty;
            if (unvisited_claimed != 0) {
                int root = first_cell(unvisited_claimed);
                visited_claimed |= cell_bit(root);
                stack.push_back(root);
            } else if (unvisited_empty != 0) {
                int root = first_cell(unvisited_empty);
                visited_empty |= cell_bit(root);
                stack.push_back(kCellCount + root);
            } else {
                break;
            }

            int reachable_size = 0;
            while (!stack.empty()) {
                int node = stack.back();
                stack.pop_back();
                if (node < kCellCount) {
                    int root = node;
                    reachable_size += sizes[root];
                    std::uint64_t linked_empty = claimed_adjacent_empty[player][root] & ~visited_empty;
                    while (linked_empty != 0) {
                        int empty_root = pop_first_cell(linked_empty);
                        visited_empty |= cell_bit(empty_root);
                        stack.push_back(kCellCount + empty_root);
                    }
                } else {
                    int root = node - kCellCount;
                    reachable_size += popcount(empty_component_cells[root]);
                    std::uint64_t linked_claimed = empty_adjacency[root][player] & ~visited_claimed;
                    while (linked_claimed != 0) {
                        int claimed_root = pop_first_cell(linked_claimed);
                        visited_claimed |= cell_bit(claimed_root);
                        stack.push_back(claimed_root);
                    }
                }
            }
            bounds.push_back(reachable_size);
        }

        std::sort(bounds.begin(), bounds.end(), std::greater<int>());
        return bounds;
    }

    std::optional<int> proven_winner() const {
        if (!selected.empty()) {
            return std::nullopt;
        }
        if (kCellCount - empty_count() < kEarlyWinCheckMinFilledCells) {
            return std::nullopt;
        }

        auto blue_sizes = group_sizes(kPlayerOne);
        auto white_sizes = group_sizes(kPlayerTwo);
        auto blue_bound = reachable_group_bounds(kPlayerOne);
        auto white_bound = reachable_group_bounds(kPlayerTwo);

        if (compare_size_vectors(blue_sizes, white_bound) > 0) {
            return kPlayerOne;
        }
        if (compare_size_vectors(white_sizes, blue_bound) > 0) {
            return kPlayerTwo;
        }
        return std::nullopt;
    }

    bool is_terminal() const {
        if (!selected.empty()) {
            return false;
        }
        int empties = empty_count();
        if (empties == 0) {
            return true;
        }
        if (kCellCount - empties < kEarlyWinCheckMinFilledCells) {
            return false;
        }
        return proven_winner().has_value();
    }

    std::optional<int> winner() const {
        auto comparison = compare_size_vectors(group_sizes(kPlayerOne), group_sizes(kPlayerTwo));
        if (comparison > 0) {
            return kPlayerOne;
        }
        if (comparison < 0) {
            return kPlayerTwo;
        }
        return std::nullopt;
    }

    int result_for(int player) const {
        auto winner_value = winner();
        if (!winner_value.has_value()) {
            return 0;
        }
        return winner_value.value() == player ? 1 : -1;
    }

    std::vector<int> legal_actions() const {
        if (is_terminal()) {
            return {};
        }

        std::vector<int> actions;
        if (!selected.empty()) {
            actions.push_back(kFinish);
        }

        if (static_cast<int>(selected.size()) < max_claims) {
            int min_cell = selected.empty() ? 0 : selected.back() + 1;
            for (int cell = std::max(0, min_cell); cell < kCellCount; ++cell) {
                if (owners[cell] == kEmpty) {
                    actions.push_back(cell);
                }
            }
        }
        return actions;
    }

    void apply_action(int action) {
        if (action == kFinish) {
            finish_turn();
            return;
        }

        claim(current_player, action);
        selected.push_back(action);
        if (static_cast<int>(selected.size()) == max_claims || empty_count() == 0) {
            finish_turn();
        }
    }

    void finish_turn() {
        int new_largest = largest_group_size();
        bool increased_global_largest = new_largest > turn_start_largest;
        int next_max_claims = (opening_turn || !increased_global_largest) ? 2 : 3;

        selected.clear();
        current_player = other_player(current_player);
        max_claims = next_max_claims;
        turn_start_largest = new_largest;
        opening_turn = false;
        ++completed_turns;
    }

private:
    void rebuild_tracking() {
        parents.fill(-1);
        sizes.fill(0);
        roots_mask.fill(0);
        max_group_size.fill(0);
        empty_component_of.fill(-1);
        empty_component_cells.fill(0);
        empty_roots_mask = 0;
        empty_count_cached = 0;
        for (auto& player_histogram : size_histogram) {
            player_histogram.fill(0);
        }
        for (auto& adjacency : empty_adjacency) {
            adjacency.fill(0);
        }
        for (auto& player_links : claimed_adjacent_empty) {
            player_links.fill(0);
        }

        std::uint64_t empty_cells = 0;
        for (int cell = 0; cell < kCellCount; ++cell) {
            int owner = owners[cell];
            if (owner == kEmpty) {
                empty_cells |= cell_bit(cell);
                ++empty_count_cached;
            } else {
                parents[cell] = cell;
                sizes[cell] = 1;
                roots_mask[owner] |= cell_bit(cell);
                add_group_size(owner, 1);
            }
        }

        const auto& neighbors = board().neighbors;
        for (int cell = 0; cell < kCellCount; ++cell) {
            int owner = owners[cell];
            if (owner == kEmpty) {
                continue;
            }
            for (int neighbor : neighbors[cell]) {
                if (neighbor > cell && owners[neighbor] == owner) {
                    union_components(owner, cell, neighbor);
                }
            }
        }

        rebuild_empty_components_from_mask(empty_cells);
    }

    void claim(int player, int cell) {
        int old_empty_root = empty_component_of[cell];
        owners[cell] = player;
        --empty_count_cached;
        parents[cell] = cell;
        sizes[cell] = 1;
        roots_mask[player] |= cell_bit(cell);
        add_group_size(player, 1);

        for (int neighbor : board().neighbors[cell]) {
            if (owners[neighbor] == player) {
                union_components(player, cell, neighbor);
            }
        }

        split_empty_region_after_claim(old_empty_root, cell);
    }

    int find_root(int cell) {
        int parent = parents[cell];
        if (parent != cell) {
            parents[cell] = find_root(parent);
        }
        return parents[cell];
    }

    int union_components(int player, int first, int second) {
        int first_root = find_root(first);
        int second_root = find_root(second);
        if (first_root == second_root) {
            return first_root;
        }

        if (sizes[first_root] < sizes[second_root]) {
            std::swap(first_root, second_root);
        }

        int first_size = sizes[first_root];
        int second_size = sizes[second_root];
        int merged_size = first_size + second_size;

        parents[second_root] = first_root;
        sizes[first_root] = merged_size;
        sizes[second_root] = 0;
        remove_group_size(player, first_size);
        remove_group_size(player, second_size);
        add_group_size(player, merged_size);
        roots_mask[player] &= ~cell_bit(second_root);
        replace_adjacent_claimed_root(player, second_root, first_root);
        return first_root;
    }

    void add_group_size(int player, int size) {
        ++size_histogram[player][size];
        max_group_size[player] = std::max(max_group_size[player], size);
    }

    void remove_group_size(int player, int size) {
        --size_histogram[player][size];
    }

    void split_empty_region_after_claim(int old_root, int claimed_cell) {
        if (old_root == -1) {
            return;
        }

        std::uint64_t old_cells = empty_component_cells[old_root];
        std::uint64_t remaining = old_cells & ~cell_bit(claimed_cell);
        if (remaining == 0) {
            unregister_empty_component(old_root);
            return;
        }

        std::uint64_t unvisited = remaining;
        EmptyFloodResult first = flood_empty_component(first_cell(unvisited), unvisited);
        if (unvisited == 0 && old_root != claimed_cell) {
            empty_component_cells[old_root] = remaining;
            empty_component_of[claimed_cell] = -1;
            refresh_empty_component_adjacency(old_root, first.adjacency);
            return;
        }

        unregister_empty_component(old_root);
        register_empty_component(first_cell(first.cells), first.cells, first.adjacency);
        rebuild_empty_components_from_mask(unvisited);
    }

    void rebuild_empty_components_from_mask(std::uint64_t cells) {
        while (cells != 0) {
            EmptyFloodResult result = flood_empty_component(first_cell(cells), cells);
            register_empty_component(first_cell(result.cells), result.cells, result.adjacency);
        }
    }

    EmptyFloodResult flood_empty_component(int start, std::uint64_t& remaining) {
        remaining &= ~cell_bit(start);
        EmptyFloodResult result;
        result.cells = cell_bit(start);
        std::vector<int> stack = {start};

        while (!stack.empty()) {
            int cell = stack.back();
            stack.pop_back();
            for (int neighbor : board().neighbors[cell]) {
                if (owners[neighbor] == kEmpty) {
                    std::uint64_t neighbor_bit = cell_bit(neighbor);
                    if ((remaining & neighbor_bit) != 0) {
                        remaining &= ~neighbor_bit;
                        result.cells |= neighbor_bit;
                        stack.push_back(neighbor);
                    }
                } else {
                    int owner = owners[neighbor];
                    result.adjacency[owner] |= cell_bit(find_root(neighbor));
                }
            }
        }
        return result;
    }

    void register_empty_component(
        int root,
        std::uint64_t cells,
        const std::array<std::uint64_t, 2>& adjacency) {
        empty_roots_mask |= cell_bit(root);
        empty_component_cells[root] = cells;
        std::uint64_t cells_to_mark = cells;
        while (cells_to_mark != 0) {
            empty_component_of[pop_first_cell(cells_to_mark)] = root;
        }
        empty_adjacency[root] = adjacency;
        add_empty_adjacency_reverse_links(root, adjacency);
    }

    void unregister_empty_component(int root) {
        std::uint64_t cells = empty_component_cells[root];
        while (cells != 0) {
            empty_component_of[pop_first_cell(cells)] = -1;
        }
        remove_empty_adjacency_reverse_links(root, empty_adjacency[root]);
        empty_adjacency[root].fill(0);
        empty_component_cells[root] = 0;
        empty_roots_mask &= ~cell_bit(root);
    }

    void refresh_empty_component_adjacency(
        int root,
        const std::array<std::uint64_t, 2>& adjacency) {
        remove_empty_adjacency_reverse_links(root, empty_adjacency[root]);
        empty_adjacency[root] = adjacency;
        add_empty_adjacency_reverse_links(root, adjacency);
    }

    void add_empty_adjacency_reverse_links(
        int empty_root,
        const std::array<std::uint64_t, 2>& adjacency) {
        for (int player : {kPlayerOne, kPlayerTwo}) {
            std::uint64_t touching_roots = adjacency[player];
            while (touching_roots != 0) {
                int touching_claimed_root = pop_first_cell(touching_roots);
                claimed_adjacent_empty[player][touching_claimed_root] |= cell_bit(empty_root);
            }
        }
    }

    void remove_empty_adjacency_reverse_links(
        int empty_root,
        const std::array<std::uint64_t, 2>& adjacency) {
        for (int player : {kPlayerOne, kPlayerTwo}) {
            std::uint64_t touching_roots = adjacency[player];
            while (touching_roots != 0) {
                int touching_claimed_root = pop_first_cell(touching_roots);
                claimed_adjacent_empty[player][touching_claimed_root] &= ~cell_bit(empty_root);
            }
        }
    }

    void replace_adjacent_claimed_root(int player, int old_root, int new_root) {
        std::uint64_t empty_roots = claimed_adjacent_empty[player][old_root];
        claimed_adjacent_empty[player][old_root] = 0;
        if (empty_roots == 0) {
            return;
        }

        claimed_adjacent_empty[player][new_root] |= empty_roots;
        std::uint64_t roots_to_update = empty_roots;
        while (roots_to_update != 0) {
            int empty_root = pop_first_cell(roots_to_update);
            empty_adjacency[empty_root][player] &= ~cell_bit(old_root);
            empty_adjacency[empty_root][player] |= cell_bit(new_root);
        }
    }
};

template <typename StateT>
struct Node {
    explicit Node(StateT node_state, Node* parent_node = nullptr, int action = -1)
        : state(std::move(node_state)),
          parent(parent_node),
          action_from_parent(action),
          untried_actions(state.legal_actions()) {}

    StateT state;
    Node* parent = nullptr;
    int action_from_parent = -1;
    std::vector<std::pair<int, Node*>> children;
    int visits = 0;
    double total_value = 0.0;
    std::vector<int> untried_actions;

    double mean_value() const {
        return visits == 0 ? 0.0 : total_value / static_cast<double>(visits);
    }
};

template <typename StateT>
class Mcts {
public:
    Mcts(int simulation_count, std::uint64_t seed)
        : simulations_(simulation_count), rng_(seed) {
        if (simulations_ <= 0) {
            throw std::runtime_error("simulations must be positive");
        }
    }

    Node<StateT>* search(const StateT& root_state) {
        nodes_.clear();
        nodes_.push_back(std::make_unique<Node<StateT>>(root_state));
        Node<StateT>* root = nodes_.back().get();

        for (int simulation = 0; simulation < simulations_; ++simulation) {
            auto path = select_and_expand(root);
            StateT terminal = rollout(path.back()->state);
            backpropagate(path, terminal);
        }
        return root;
    }

private:
    std::vector<Node<StateT>*> select_and_expand(Node<StateT>* root) {
        std::vector<Node<StateT>*> path = {root};
        Node<StateT>* node = root;

        while (!node->state.is_terminal() && node->untried_actions.empty() && !node->children.empty()) {
            node = select_child(node);
            path.push_back(node);
        }

        if (node->state.is_terminal() || node->untried_actions.empty()) {
            return path;
        }

        int action = pop_random_untried_action(node);
        StateT child_state = node->state;
        child_state.apply_action(action);
        nodes_.push_back(std::make_unique<Node<StateT>>(std::move(child_state), node, action));
        Node<StateT>* child = nodes_.back().get();
        node->children.push_back({action, child});
        path.push_back(child);
        return path;
    }

    Node<StateT>* select_child(Node<StateT>* node) {
        const int parent_visits = std::max(node->visits, 1);
        double best_score = -std::numeric_limits<double>::infinity();
        std::vector<Node<StateT>*> best_children;

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

    int pop_random_untried_action(Node<StateT>* node) {
        std::uniform_int_distribution<std::size_t> distribution(0, node->untried_actions.size() - 1);
        std::size_t index = distribution(rng_);
        int action = node->untried_actions[index];
        node->untried_actions.erase(node->untried_actions.begin() + static_cast<std::ptrdiff_t>(index));
        return action;
    }

    StateT rollout(const StateT& state) {
        StateT current = state;
        while (!current.is_terminal()) {
            auto actions = current.legal_actions();
            std::uniform_int_distribution<std::size_t> distribution(0, actions.size() - 1);
            current.apply_action(actions[distribution(rng_)]);
        }
        return current;
    }

    static void backpropagate(const std::vector<Node<StateT>*>& path, const StateT& terminal) {
        for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
            Node<StateT>* node = *iterator;
            ++node->visits;
            node->total_value += terminal.result_for(node->state.current_player);
        }
    }

    static double mean_value_for_player(const Node<StateT>* node, int player) {
        double value = node->mean_value();
        if (node->state.current_player == player) {
            return value;
        }
        return -value;
    }

    int simulations_;
    std::mt19937_64 rng_;
    std::vector<std::unique_ptr<Node<StateT>>> nodes_;
};

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    if (text.empty()) {
        return values;
    }

    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stoi(item));
        }
    }
    return values;
}

std::string require_arg(const std::unordered_map<std::string, std::string>& args, const std::string& name) {
    auto found = args.find(name);
    if (found == args.end()) {
        throw std::runtime_error("missing argument: " + name);
    }
    return found->second;
}

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index) {
        std::string name = argv[index];
        if (name.rfind("--", 0) != 0) {
            throw std::runtime_error("unexpected positional argument: " + name);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + name);
        }
        args[name.substr(2)] = argv[++index];
    }
    return args;
}

TrackedState state_from_args(const std::unordered_map<std::string, std::string>& args) {
    TrackedState state;
    auto owners = parse_int_list(require_arg(args, "owners"));
    if (owners.size() != kCellCount) {
        throw std::runtime_error("owners must contain 61 cells");
    }
    for (int index = 0; index < kCellCount; ++index) {
        int owner = owners[index];
        if (owner != kEmpty && owner != kPlayerOne && owner != kPlayerTwo) {
            throw std::runtime_error("invalid owner value");
        }
        state.owners[index] = owner;
    }

    state.current_player = std::stoi(require_arg(args, "current-player"));
    state.selected = parse_int_list(require_arg(args, "selected"));
    state.max_claims = std::stoi(require_arg(args, "max-claims"));
    state.turn_start_largest = std::stoi(require_arg(args, "turn-start-largest"));
    state.opening_turn = std::stoi(require_arg(args, "opening-turn")) != 0;
    state.completed_turns = std::stoi(require_arg(args, "completed-turns"));
    state.rebuild_from_owners();
    return state;
}

template <typename StateT>
std::vector<std::pair<int, Node<StateT>*>> sorted_choices(const Node<StateT>* root) {
    auto choices = root->children;
    std::sort(
        choices.begin(),
        choices.end(),
        [](const auto& first, const auto& second) {
            if (first.second->visits != second.second->visits) {
                return first.second->visits > second.second->visits;
            }
            return first.first < second.first;
        });
    return choices;
}

template <typename StateT>
void print_result(const Node<StateT>* root, int simulations) {
    auto choices = sorted_choices(root);
    if (choices.empty()) {
        throw std::runtime_error("search produced no choices");
    }

    std::cout << "{";
    std::cout << "\"action\":" << choices.front().first << ",";
    std::cout << "\"player\":" << root->state.current_player << ",";
    std::cout << "\"simulations\":" << simulations << ",";
    std::cout << "\"state_mode\":\"tracked\",";
    std::cout << "\"choices\":[";
    for (std::size_t index = 0; index < choices.size(); ++index) {
        auto [action, child] = choices[index];
        if (index != 0) {
            std::cout << ",";
        }
        double value = child->visits == 0 ? 0.0 : child->mean_value();
        if (child->state.current_player != root->state.current_player) {
            value = -value;
        }
        std::cout << "{";
        std::cout << "\"action\":" << action << ",";
        std::cout << "\"visits\":" << child->visits << ",";
        std::cout << "\"value\":" << value;
        std::cout << "}";
    }
    std::cout << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        int simulations = std::stoi(require_arg(args, "simulations"));
        std::uint64_t seed = 1;
        auto seed_arg = args.find("seed");
        if (seed_arg != args.end()) {
            seed = static_cast<std::uint64_t>(std::stoull(seed_arg->second));
        }
        if (args.find("state-mode") != args.end()) {
            throw std::runtime_error("state-mode is no longer supported; tracked is the only implementation");
        }

        TrackedState state = state_from_args(args);
        Mcts<TrackedState> mcts(simulations, seed);
        Node<TrackedState>* root = mcts.search(state);
        print_result(root, simulations);
        std::cout << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
