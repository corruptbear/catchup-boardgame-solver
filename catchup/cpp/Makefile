CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

BUILD_DIR := build
TARGET := $(BUILD_DIR)/catchup_mcts
SOURCES := catchup_mcts.cpp board.cpp tracked_state.cpp mcts.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

clean:
	rm -rf $(BUILD_DIR)
