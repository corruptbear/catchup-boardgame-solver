#include "board.hpp"
#include "mcts.hpp"
#include "tracked_state.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

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

std::string require_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& name) {
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
        state.owners[index] = static_cast<Owner>(owner);
    }

    state.current_player = static_cast<Owner>(std::stoi(require_arg(args, "current-player")));
    state.selected.clear();
    for (int cell : parse_int_list(require_arg(args, "selected"))) {
        state.selected.push_back(cell);
    }
    state.max_claims = static_cast<std::uint8_t>(std::stoi(require_arg(args, "max-claims")));
    state.turn_start_largest = static_cast<std::uint8_t>(
        std::stoi(require_arg(args, "turn-start-largest")));
    state.opening_turn = std::stoi(require_arg(args, "opening-turn")) != 0;
    state.completed_turns = static_cast<std::uint8_t>(
        std::stoi(require_arg(args, "completed-turns")));
    state.rebuild_from_owners();
    return state;
}

std::vector<std::pair<int, Node*>> sorted_choices(const Node* root) {
    std::vector<std::pair<int, Node*>> choices;
    choices.reserve(root->children.size());
    for (const auto& [action, child] : root->children) {
        choices.push_back({
            static_cast<int>(action),
            child,
        });
    }
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
    std::cout << "\"player\":" << static_cast<int>(root->state.current_player) << ",";
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
        TrackedState state = state_from_args(args);
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
