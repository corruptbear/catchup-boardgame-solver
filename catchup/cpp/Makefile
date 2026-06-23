CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic
THREAD_FLAGS ?= -pthread

BUILD_DIR := build
MCTS_TARGET := $(BUILD_DIR)/catchup_mcts
ARENA_TARGET := $(BUILD_DIR)/catchup_arena
COMMON_SOURCES := board.cpp tracked_state.cpp mcts.cpp puct_mcts.cpp

.PHONY: all clean

all: $(MCTS_TARGET) $(ARENA_TARGET)

$(MCTS_TARGET): catchup_mcts.cpp $(COMMON_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) catchup_mcts.cpp $(COMMON_SOURCES) -o $@

$(ARENA_TARGET): arena.cpp $(COMMON_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(THREAD_FLAGS) arena.cpp $(COMMON_SOURCES) -o $@

clean:
	rm -rf $(BUILD_DIR)
