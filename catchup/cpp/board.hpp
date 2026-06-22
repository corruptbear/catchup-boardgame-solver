#pragma once

#include <array>
#include <string>
#include <vector>

constexpr int kEmpty = -1;
constexpr int kPlayerOne = 0;
constexpr int kPlayerTwo = 1;
constexpr int kCellCount = 61;
constexpr int kFinish = kCellCount;
constexpr int kEarlyWinCheckMinFilledCells = 30;

struct Board {
    std::array<std::vector<int>, kCellCount> neighbors;

    Board();

    static std::string key(int q, int r);
};

const Board& board();
int other_player(int player);
int compare_size_vectors(const std::vector<int>& first, const std::vector<int>& second);
