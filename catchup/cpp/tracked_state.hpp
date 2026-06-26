#pragma once

#include "board.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

struct ActionList {
    std::array<Action, kMaxActions> values{};
    std::uint8_t count = 0;

    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }
    Action operator[](std::size_t index) const { return values[index]; }
    Action back() const { return values[count - 1]; }
    void clear() { count = 0; }

    void push_back(int action) {
        if (count >= values.size()) {
            throw std::runtime_error("too many actions");
        }
        values[count++] = static_cast<Action>(action);
    }

    void erase_unordered(std::size_t index) {
        values[index] = values[--count];
    }
};

struct SelectedCells {
    std::array<Action, kMaxClaims> values{};
    std::uint8_t count = 0;

    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }
    Action operator[](std::size_t index) const { return values[index]; }
    Action back() const { return values[count - 1]; }
    void clear() { count = 0; }

    void push_back(int cell) {
        if (count >= values.size()) {
            throw std::runtime_error("too many selected cells");
        }
        values[count++] = static_cast<Action>(cell);
    }
};

struct TrackedState {
    std::array<Owner, kCellCount> owners{};
    std::array<std::int8_t, kCellCount> parents{};
    std::array<std::uint8_t, kCellCount> sizes{};
    std::array<std::uint64_t, 2> roots_mask{};
    std::array<std::array<std::uint8_t, kCellCount + 1>, 2> size_histogram{};
    std::array<std::uint8_t, 2> max_group_size{};
    std::uint8_t empty_count_cached = kCellCount;
    std::array<std::int8_t, kCellCount> empty_component_of{};
    std::array<std::uint64_t, kCellCount> empty_component_cells{};
    std::uint64_t empty_roots_mask = 0;
    std::array<std::uint64_t, kCellCount> region_neighbors{};
    Owner current_player = kPlayerOne;
    SelectedCells selected;
    std::uint8_t max_claims = 1;
    std::uint8_t turn_start_largest = 0;
    bool opening_turn = true;
    bool early_win_enabled = true;
    std::uint8_t completed_turns = 0;

    TrackedState();

    void rebuild_from_owners();
    int empty_count() const;
    std::vector<int> group_sizes(int player) const;
    int largest_group_size() const;
    std::vector<int> reachable_group_bounds(int player) const;
    std::optional<int> proven_winner() const;
    bool is_terminal() const;
    int winner() const;
    int result_for(int player) const;
    ActionList legal_actions() const;
    void apply_action(int action);
    void finish_turn();

private:
    struct EmptyFloodResult {
        std::uint64_t cells = 0;
        std::uint64_t touching_claimed = 0;
    };

    std::array<std::uint8_t, kCellCount + 1> reachable_group_bound_histogram(
        int player) const;
    void rebuild_tracking();
    void claim(int player, int cell);
    int find_root(int cell);
    int union_components(int player, int first, int second);
    void add_group_size(int player, int size);
    void remove_group_size(int player, int size);
    void split_empty_region_after_claim(int old_root, int claimed_cell);
    void rebuild_empty_components_from_mask(std::uint64_t cells);
    EmptyFloodResult flood_empty_component(int start, std::uint64_t& remaining);
    void register_empty_component(
        int root,
        std::uint64_t cells,
        std::uint64_t touching_claimed);
    void unregister_empty_component(int root);
    void refresh_empty_region_neighbors(
        int root,
        std::uint64_t touching_claimed);
    void add_region_edges(int root, std::uint64_t neighbors);
    void remove_region_edges(int root, std::uint64_t neighbors);
    void replace_region_root(int old_root, int new_root);
    void add_opponent_claimed_neighbors(int player, int root, int cell);
};
