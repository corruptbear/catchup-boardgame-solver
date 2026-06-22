#include "tracked_state.hpp"

#include <algorithm>
#include <functional>

namespace {

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

constexpr int kMaxRegionsPerOwner = 21;
constexpr int kMaxReachableRegionNodes = 2 * kMaxRegionsPerOwner;
using SizeHistogram = std::array<std::uint8_t, kCellCount + 1>;

int compare_size_histograms(
    const SizeHistogram& first,
    const SizeHistogram& second) {
    for (int size = kCellCount; size > 0; --size) {
        if (first[size] != second[size]) {
            return first[size] > second[size] ? 1 : -1;
        }
    }
    return 0;
}

}  // namespace

TrackedState::TrackedState() {
    owners.fill(kEmpty);
    rebuild_tracking();
}

void TrackedState::rebuild_from_owners() {
    rebuild_tracking();
}

int TrackedState::empty_count() const {
    return empty_count_cached;
}

std::vector<int> TrackedState::group_sizes(int player) const {
    std::vector<int> result;
    result.reserve(kMaxRegionsPerOwner);
    for (int size = max_group_size[player]; size > 0; --size) {
        int count = size_histogram[player][size];
        for (int index = 0; index < count; ++index) {
            result.push_back(size);
        }
    }
    return result;
}

int TrackedState::largest_group_size() const {
    return std::max(max_group_size[kPlayerOne], max_group_size[kPlayerTwo]);
}

std::vector<int> TrackedState::reachable_group_bounds(int player) const {
    auto histogram = reachable_group_bound_histogram(player);
    std::vector<int> bounds;
    bounds.reserve(kMaxReachableRegionNodes);
    for (int size = kCellCount; size > 0; --size) {
        int count = histogram[size];
        for (int index = 0; index < count; ++index) {
            bounds.push_back(size);
        }
    }
    return bounds;
}

SizeHistogram TrackedState::reachable_group_bound_histogram(int player) const {
    SizeHistogram bounds{};
    std::uint64_t visited_claimed = 0;
    std::uint64_t visited_empty = 0;

    while (true) {
        std::array<int, kMaxReachableRegionNodes> stack{};
        int stack_size = 0;
        std::uint64_t unvisited_claimed = roots_mask[player] & ~visited_claimed;
        std::uint64_t unvisited_empty = empty_roots_mask & ~visited_empty;
        if (unvisited_claimed != 0) {
            int root = first_cell(unvisited_claimed);
            visited_claimed |= cell_bit(root);
            stack[stack_size++] = root;
        } else if (unvisited_empty != 0) {
            int root = first_cell(unvisited_empty);
            visited_empty |= cell_bit(root);
            stack[stack_size++] = kCellCount + root;
        } else {
            break;
        }

        int reachable_size = 0;
        while (stack_size != 0) {
            int node = stack[--stack_size];
            if (node < kCellCount) {
                int root = node;
                reachable_size += sizes[root];
                std::uint64_t linked_empty =
                    region_neighbors[root] & empty_roots_mask & ~visited_empty;
                while (linked_empty != 0) {
                    int empty_root = pop_first_cell(linked_empty);
                    visited_empty |= cell_bit(empty_root);
                    stack[stack_size++] = kCellCount + empty_root;
                }
            } else {
                int root = node - kCellCount;
                reachable_size += popcount(empty_component_cells[root]);
                std::uint64_t linked_claimed =
                    region_neighbors[root] & roots_mask[player] & ~visited_claimed;
                while (linked_claimed != 0) {
                    int claimed_root = pop_first_cell(linked_claimed);
                    visited_claimed |= cell_bit(claimed_root);
                    stack[stack_size++] = claimed_root;
                }
            }
        }
        ++bounds[reachable_size];
    }
    return bounds;
}

std::optional<int> TrackedState::proven_winner() const {
    if (!selected.empty()) {
        return std::nullopt;
    }
    if (kCellCount - empty_count() < kEarlyWinCheckMinFilledCells) {
        return std::nullopt;
    }

    auto blue_bound_histogram = reachable_group_bound_histogram(kPlayerOne);
    auto white_bound_histogram = reachable_group_bound_histogram(kPlayerTwo);

    if (compare_size_histograms(size_histogram[kPlayerOne], white_bound_histogram) > 0) {
        return kPlayerOne;
    }
    if (compare_size_histograms(size_histogram[kPlayerTwo], blue_bound_histogram) > 0) {
        return kPlayerTwo;
    }
    return std::nullopt;
}

bool TrackedState::is_terminal() const {
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

std::optional<int> TrackedState::winner() const {
    auto comparison = compare_size_histograms(
        size_histogram[kPlayerOne],
        size_histogram[kPlayerTwo]);
    if (comparison > 0) {
        return kPlayerOne;
    }
    if (comparison < 0) {
        return kPlayerTwo;
    }
    return std::nullopt;
}

int TrackedState::result_for(int player) const {
    auto winner_value = winner();
    if (!winner_value.has_value()) {
        return 0;
    }
    return winner_value.value() == player ? 1 : -1;
}

ActionList TrackedState::legal_actions() const {
    if (is_terminal()) {
        return {};
    }

    ActionList actions;
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

void TrackedState::apply_action(int action) {
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

void TrackedState::finish_turn() {
    int new_largest = largest_group_size();
    bool increased_global_largest = new_largest > turn_start_largest;
    int next_max_claims = (opening_turn || !increased_global_largest) ? 2 : 3;

    selected.clear();
    current_player = static_cast<Owner>(other_player(current_player));
    max_claims = static_cast<std::uint8_t>(next_max_claims);
    turn_start_largest = static_cast<std::uint8_t>(new_largest);
    opening_turn = false;
    ++completed_turns;
}

void TrackedState::rebuild_tracking() {
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
    region_neighbors.fill(0);

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

void TrackedState::claim(int player, int cell) {
    int old_empty_root = empty_component_of[cell];
    if (old_empty_root != -1) {
        remove_region_edges(old_empty_root, region_neighbors[old_empty_root]);
        region_neighbors[old_empty_root] = 0;
    }

    owners[cell] = static_cast<Owner>(player);
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
    add_opponent_claimed_neighbors(player, find_root(cell), cell);
}

int TrackedState::find_root(int cell) {
    int parent = parents[cell];
    if (parent != cell) {
        parents[cell] = find_root(parent);
    }
    return parents[cell];
}

int TrackedState::union_components(int player, int first, int second) {
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
    replace_region_root(second_root, first_root);
    return first_root;
}

void TrackedState::add_group_size(int player, int size) {
    ++size_histogram[player][size];
    max_group_size[player] = static_cast<std::uint8_t>(
        std::max<int>(max_group_size[player], size));
}

void TrackedState::remove_group_size(int player, int size) {
    --size_histogram[player][size];
}

void TrackedState::split_empty_region_after_claim(int old_root, int claimed_cell) {
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
        refresh_empty_region_neighbors(old_root, first.touching_claimed);
        return;
    }

    unregister_empty_component(old_root);
    register_empty_component(first_cell(first.cells), first.cells, first.touching_claimed);
    rebuild_empty_components_from_mask(unvisited);
}

void TrackedState::rebuild_empty_components_from_mask(std::uint64_t cells) {
    while (cells != 0) {
        EmptyFloodResult result = flood_empty_component(first_cell(cells), cells);
        register_empty_component(first_cell(result.cells), result.cells, result.touching_claimed);
    }
}

TrackedState::EmptyFloodResult TrackedState::flood_empty_component(
    int start,
    std::uint64_t& remaining) {
    remaining &= ~cell_bit(start);
    EmptyFloodResult result;
    result.cells = cell_bit(start);
    std::array<int, kCellCount> stack{};
    int stack_size = 0;
    stack[stack_size++] = start;

    while (stack_size != 0) {
        int cell = stack[--stack_size];
        for (int neighbor : board().neighbors[cell]) {
            if (owners[neighbor] == kEmpty) {
                std::uint64_t neighbor_bit = cell_bit(neighbor);
                if ((remaining & neighbor_bit) != 0) {
                    remaining &= ~neighbor_bit;
                    result.cells |= neighbor_bit;
                    stack[stack_size++] = neighbor;
                }
            } else {
                result.touching_claimed |= cell_bit(find_root(neighbor));
            }
        }
    }
    return result;
}

void TrackedState::register_empty_component(
    int root,
    std::uint64_t cells,
    std::uint64_t touching_claimed) {
    empty_roots_mask |= cell_bit(root);
    empty_component_cells[root] = cells;
    std::uint64_t cells_to_mark = cells;
    while (cells_to_mark != 0) {
        empty_component_of[pop_first_cell(cells_to_mark)] = root;
    }
    region_neighbors[root] = 0;
    add_region_edges(root, touching_claimed);
}

void TrackedState::unregister_empty_component(int root) {
    std::uint64_t cells = empty_component_cells[root];
    while (cells != 0) {
        empty_component_of[pop_first_cell(cells)] = -1;
    }
    remove_region_edges(root, region_neighbors[root]);
    region_neighbors[root] = 0;
    empty_component_cells[root] = 0;
    empty_roots_mask &= ~cell_bit(root);
}

void TrackedState::refresh_empty_region_neighbors(
    int root,
    std::uint64_t touching_claimed) {
    remove_region_edges(root, region_neighbors[root]);
    region_neighbors[root] = 0;
    add_region_edges(root, touching_claimed);
}

void TrackedState::add_region_edges(int root, std::uint64_t neighbors) {
    neighbors &= ~cell_bit(root);
    region_neighbors[root] |= neighbors;
    while (neighbors != 0) {
        int neighbor_root = pop_first_cell(neighbors);
        region_neighbors[neighbor_root] |= cell_bit(root);
    }
}

void TrackedState::remove_region_edges(int root, std::uint64_t neighbors) {
    while (neighbors != 0) {
        int neighbor_root = pop_first_cell(neighbors);
        region_neighbors[neighbor_root] &= ~cell_bit(root);
    }
}

void TrackedState::replace_region_root(int old_root, int new_root) {
    std::uint64_t neighbors = region_neighbors[old_root];
    region_neighbors[old_root] = 0;
    region_neighbors[new_root] &= ~cell_bit(old_root);

    neighbors &= ~cell_bit(new_root);
    if (neighbors == 0) {
        return;
    }

    region_neighbors[new_root] |= neighbors;
    region_neighbors[new_root] &= ~cell_bit(new_root);
    std::uint64_t roots_to_update = neighbors;
    while (roots_to_update != 0) {
        int neighbor_root = pop_first_cell(roots_to_update);
        region_neighbors[neighbor_root] &= ~cell_bit(old_root);
        region_neighbors[neighbor_root] |= cell_bit(new_root);
    }
}

void TrackedState::add_opponent_claimed_neighbors(int player, int root, int cell) {
    std::uint64_t neighbors = 0;
    for (int neighbor : board().neighbors[cell]) {
        int owner = owners[neighbor];
        if (owner != kEmpty && owner != player) {
            neighbors |= cell_bit(find_root(neighbor));
        }
    }
    add_region_edges(root, neighbors);
}
