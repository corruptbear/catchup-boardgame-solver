CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic

BUILD_DIR := build
TARGET := $(BUILD_DIR)/catchup_mcts
SOURCE := catchup_mcts.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
