CXX ?= c++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -pedantic
THREAD_FLAGS ?= -pthread
PYTHON ?= python3.10
TORCH_INCLUDE_FLAGS ?= $(shell $(PYTHON) -c 'import torch.utils.cpp_extension as ext; print(" ".join("-isystem " + path for path in ext.include_paths()))')
TORCH_LIBRARY_FLAGS ?= $(shell $(PYTHON) -c 'import torch.utils.cpp_extension as ext; print(" ".join("-L" + path + " -Wl,-rpath," + path for path in ext.library_paths()))') -ltorch -ltorch_cpu -lc10
MLX_ROOT ?= $(shell $(PYTHON) -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("mlx").submodule_search_locations[0]))')
MLX_INCLUDE_FLAGS ?= -isystem $(MLX_ROOT)/include -isystem $(MLX_ROOT)/include/metal_cpp
MLX_LIBRARY_FLAGS ?= -L$(MLX_ROOT)/lib -Wl,-rpath,$(MLX_ROOT)/lib -lmlx -ljaccl -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate

BUILD_DIR := build
MCTS_TARGET := $(BUILD_DIR)/catchup_mcts
ARENA_TARGET := $(BUILD_DIR)/catchup_arena
SELF_PLAY_TARGET := $(BUILD_DIR)/catchup_self_play
COMMON_SOURCES := board.cpp tracked_state.cpp mcts.cpp puct_mcts.cpp
NEURAL_SOURCES := puct_neural.cpp inference_aoti.cpp inference_mlx.cpp

.PHONY: all clean

all: $(MCTS_TARGET) $(ARENA_TARGET) $(SELF_PLAY_TARGET)

$(MCTS_TARGET): catchup_mcts.cpp $(COMMON_SOURCES) $(NEURAL_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TORCH_INCLUDE_FLAGS) $(MLX_INCLUDE_FLAGS) catchup_mcts.cpp $(COMMON_SOURCES) $(NEURAL_SOURCES) $(TORCH_LIBRARY_FLAGS) $(MLX_LIBRARY_FLAGS) -o $@

$(ARENA_TARGET): arena.cpp $(COMMON_SOURCES) $(NEURAL_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(THREAD_FLAGS) $(TORCH_INCLUDE_FLAGS) $(MLX_INCLUDE_FLAGS) arena.cpp $(COMMON_SOURCES) $(NEURAL_SOURCES) $(TORCH_LIBRARY_FLAGS) $(MLX_LIBRARY_FLAGS) -o $@

$(SELF_PLAY_TARGET): self_play.cpp $(COMMON_SOURCES) $(NEURAL_SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(THREAD_FLAGS) $(TORCH_INCLUDE_FLAGS) $(MLX_INCLUDE_FLAGS) self_play.cpp $(COMMON_SOURCES) $(NEURAL_SOURCES) $(TORCH_LIBRARY_FLAGS) $(MLX_LIBRARY_FLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)
