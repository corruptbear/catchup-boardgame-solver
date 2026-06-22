#pragma once

#include "board.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

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

    TrackedState();

    void rebuild_from_owners();
    int empty_count() const;
    std::vector<int> group_sizes(int player) const;
    int largest_group_size() const;
    std::vector<int> reachable_group_bounds(int player) const;
    std::optional<int> proven_winner() const;
    bool is_terminal() const;
    std::optional<int> winner() const;
    int result_for(int player) const;
    std::vector<int> legal_actions() const;
    void apply_action(int action);
    void finish_turn();

private:
    struct EmptyFloodResult {
        std::uint64_t cells = 0;
        std::array<std::uint64_t, 2> adjacency{};
    };

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
        const std::array<std::uint64_t, 2>& adjacency);
    void unregister_empty_component(int root);
    void refresh_empty_component_adjacency(
        int root,
        const std::array<std::uint64_t, 2>& adjacency);
    void add_empty_adjacency_reverse_links(
        int empty_root,
        const std::array<std::uint64_t, 2>& adjacency);
    void remove_empty_adjacency_reverse_links(
        int empty_root,
        const std::array<std::uint64_t, 2>& adjacency);
    void replace_adjacent_claimed_root(int player, int old_root, int new_root);
};
