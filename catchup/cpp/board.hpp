#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using Action = std::uint8_t;
using Owner = std::int8_t;

constexpr Owner kEmpty = -1;
constexpr Owner kPlayerOne = 0;
constexpr Owner kPlayerTwo = 1;
constexpr int kCellCount = 61;
constexpr int kMaxActions = kCellCount + 1;
constexpr int kMaxClaims = 3;
constexpr Action kFinish = kCellCount;
constexpr int kEarlyWinCheckMinFilledCells = 30;

struct Board {
    std::array<std::pair<int, int>, kCellCount> coords;
    std::array<std::vector<int>, kCellCount> neighbors;

    Board();

    static std::string key(int q, int r);
};

const Board& board();
int other_player(int player);
int compare_size_vectors(const std::vector<int>& first, const std::vector<int>& second);
