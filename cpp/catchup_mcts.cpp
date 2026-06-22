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

struct State {
    std::array<int, kCellCount> owners{};
    int current_player = kPlayerOne;
    std::vector<int> selected;
    int max_claims = 1;
    int turn_start_largest = 0;
    bool opening_turn = true;
    int completed_turns = 0;

    State() {
        owners.fill(kEmpty);
    }

    int empty_count() const {
        int count = 0;
        for (int owner : owners) {
            if (owner == kEmpty) {
                ++count;
            }
        }
        return count;
    }

    std::vector<int> group_sizes(int player) const {
        std::array<bool, kCellCount> visited{};
        std::vector<int> sizes;
        const auto& neighbors = board().neighbors;

        for (int cell = 0; cell < kCellCount; ++cell) {
            if (visited[cell] || owners[cell] != player) {
                continue;
            }

            int size = 0;
            std::vector<int> stack = {cell};
            visited[cell] = true;
            while (!stack.empty()) {
                int current = stack.back();
                stack.pop_back();
                ++size;
                for (int neighbor : neighbors[current]) {
                    if (!visited[neighbor] && owners[neighbor] == player) {
                        visited[neighbor] = true;
                        stack.push_back(neighbor);
                    }
                }
            }
            sizes.push_back(size);
        }

        std::sort(sizes.begin(), sizes.end(), std::greater<int>());
        return sizes;
    }

    int largest_group_size() const {
        int largest = 0;
        for (int player : {kPlayerOne, kPlayerTwo}) {
            auto sizes = group_sizes(player);
            if (!sizes.empty()) {
                largest = std::max(largest, sizes.front());
            }
        }
        return largest;
    }

    std::vector<int> reachable_group_bounds(int player) const {
        std::array<bool, kCellCount> visited{};
        std::vector<int> bounds;
        const int opponent = other_player(player);
        const auto& neighbors = board().neighbors;

        for (int cell = 0; cell < kCellCount; ++cell) {
            if (visited[cell] || owners[cell] == opponent) {
                continue;
            }

            int size = 0;
            std::vector<int> stack = {cell};
            visited[cell] = true;
            while (!stack.empty()) {
                int current = stack.back();
                stack.pop_back();
                ++size;
                for (int neighbor : neighbors[current]) {
                    if (!visited[neighbor] && owners[neighbor] != opponent) {
                        visited[neighbor] = true;
                        stack.push_back(neighbor);
                    }
                }
            }
            bounds.push_back(size);
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

        owners[action] = current_player;
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
};

struct Node {
    explicit Node(State node_state, Node* parent_node = nullptr, int action = -1)
        : state(std::move(node_state)),
          parent(parent_node),
          action_from_parent(action),
          untried_actions(state.legal_actions()) {}

    State state;
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

class Mcts {
public:
    Mcts(int simulation_count, std::uint64_t seed)
        : simulations_(simulation_count), rng_(seed) {
        if (simulations_ <= 0) {
            throw std::runtime_error("simulations must be positive");
        }
    }

    Node* search(const State& root_state) {
        nodes_.clear();
        nodes_.push_back(std::make_unique<Node>(root_state));
        Node* root = nodes_.back().get();

        for (int simulation = 0; simulation < simulations_; ++simulation) {
            auto path = select_and_expand(root);
            State terminal = rollout(path.back()->state);
            backpropagate(path, terminal);
        }
        return root;
    }

private:
    std::vector<Node*> select_and_expand(Node* root) {
        std::vector<Node*> path = {root};
        Node* node = root;

        while (!node->state.is_terminal() && node->untried_actions.empty() && !node->children.empty()) {
            node = select_child(node);
            path.push_back(node);
        }

        if (node->state.is_terminal() || node->untried_actions.empty()) {
            return path;
        }

        int action = pop_random_untried_action(node);
        State child_state = node->state;
        child_state.apply_action(action);
        nodes_.push_back(std::make_unique<Node>(std::move(child_state), node, action));
        Node* child = nodes_.back().get();
        node->children.push_back({action, child});
        path.push_back(child);
        return path;
    }

    Node* select_child(Node* node) {
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

    int pop_random_untried_action(Node* node) {
        std::uniform_int_distribution<std::size_t> distribution(0, node->untried_actions.size() - 1);
        std::size_t index = distribution(rng_);
        int action = node->untried_actions[index];
        node->untried_actions.erase(node->untried_actions.begin() + static_cast<std::ptrdiff_t>(index));
        return action;
    }

    State rollout(const State& state) {
        State current = state;
        while (!current.is_terminal()) {
            auto actions = current.legal_actions();
            std::uniform_int_distribution<std::size_t> distribution(0, actions.size() - 1);
            current.apply_action(actions[distribution(rng_)]);
        }
        return current;
    }

    static void backpropagate(const std::vector<Node*>& path, const State& terminal) {
        for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
            Node* node = *iterator;
            ++node->visits;
            node->total_value += terminal.result_for(node->state.current_player);
        }
    }

    static double mean_value_for_player(const Node* node, int player) {
        double value = node->mean_value();
        if (node->state.current_player == player) {
            return value;
        }
        return -value;
    }

    int simulations_;
    std::mt19937_64 rng_;
    std::vector<std::unique_ptr<Node>> nodes_;
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

State state_from_args(const std::unordered_map<std::string, std::string>& args) {
    State state;
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
    return state;
}

std::vector<std::pair<int, Node*>> sorted_choices(const Node* root) {
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

void print_result(const Node* root, int simulations) {
    auto choices = sorted_choices(root);
    if (choices.empty()) {
        throw std::runtime_error("search produced no choices");
    }

    std::cout << "{";
    std::cout << "\"action\":" << choices.front().first << ",";
    std::cout << "\"player\":" << root->state.current_player << ",";
    std::cout << "\"simulations\":" << simulations << ",";
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

        State state = state_from_args(args);
        Mcts mcts(simulations, seed);
        Node* root = mcts.search(state);
        print_result(root, simulations);
        std::cout << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
