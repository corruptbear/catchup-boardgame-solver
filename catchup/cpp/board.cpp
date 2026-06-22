#include "board.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

Board::Board() {
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

std::string Board::key(int q, int r) {
    return std::to_string(q) + "," + std::to_string(r);
}

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
