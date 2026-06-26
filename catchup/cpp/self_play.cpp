#include "puct_mcts.hpp"
#include "puct_neural.hpp"
#include "tracked_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

enum class TeacherKind {
    Puct,
    NeuralPuct,
};

struct Config {
    int games = 1;
    int simulations = 1000;
    std::uint64_t seed = 0;
    int threads = 1;
    int max_actions = 512;
    TeacherKind teacher = TeacherKind::Puct;
    PuctConfig puct_config;
    std::string model_path;
    NeuralBackend neural_backend = NeuralBackend::Aoti;
    NeuralDevice neural_device = NeuralDevice::Mps;
    int neural_batch_size = 32;
    double neural_batch_wait_ms = 0.5;
    NeuralPuctConfig neural_puct_config;
    double visit_temperature_min = 0.05;
    std::string output_path;
    bool profile_neural_batch = false;
};

struct TerminalInfo {
    int winner = kPlayerOne;
    std::vector<int> blue_group_sizes;
    std::vector<int> white_group_sizes;
    int filled_cells = 0;
    int completed_turns = 0;
};

struct Sample {
    TrackedState state;
    std::vector<double> policy_target;
    TerminalInfo terminal;
    int game_id = 0;
    int ply = 0;
    std::uint64_t search_seed = 0;
    int chosen_action = 0;
    int value_target = 0;
};

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index) {
        std::string name = argv[index];
        if (name.rfind("--", 0) != 0) {
            throw std::runtime_error("unexpected positional argument: " + name);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + name);
        }
        args[name.substr(2)] = argv[++index];
    }
    return args;
}

std::string require_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& name) {
    auto found = args.find(name);
    if (found == args.end()) {
        throw std::runtime_error("missing argument: " + name);
    }
    return found->second;
}

std::string arg_or_default(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& name,
    const std::string& default_value) {
    auto found = args.find(name);
    return found == args.end() ? default_value : found->second;
}

std::uint64_t random_seed() {
    std::random_device device;
    std::uint64_t high = static_cast<std::uint64_t>(device()) << 32;
    return high ^ static_cast<std::uint64_t>(device());
}

Config parse_config(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    Config config;
    config.games = std::stoi(require_arg(args, "games"));
    config.simulations = std::stoi(require_arg(args, "simulations"));
    config.output_path = require_arg(args, "out");
    auto seed_arg = args.find("seed");
    config.seed = seed_arg == args.end()
        ? random_seed()
        : static_cast<std::uint64_t>(std::stoull(seed_arg->second));
    config.threads = std::stoi(arg_or_default(
        args,
        "threads",
        std::to_string(std::max(1u, std::thread::hardware_concurrency()))));
    config.max_actions = std::stoi(arg_or_default(args, "max-actions", "512"));
    std::string teacher = arg_or_default(args, "teacher", "puct");
    if (teacher == "puct") {
        config.teacher = TeacherKind::Puct;
    } else if (teacher == "neural-puct") {
        config.teacher = TeacherKind::NeuralPuct;
        config.model_path = require_arg(args, "model");
    } else {
        throw std::runtime_error("teacher must be puct or neural-puct");
    }
    config.puct_config.prior = parse_puct_prior_mode(
        arg_or_default(args, "puct-prior", "heuristic"));
    config.puct_config.rollout = parse_puct_rollout_mode(
        arg_or_default(args, "puct-rollout", "biased"));
    config.neural_batch_size = std::stoi(arg_or_default(args, "neural-batch-size", "32"));
    config.neural_batch_wait_ms = std::stod(arg_or_default(args, "neural-batch-wait-ms", "0.5"));
    config.neural_backend = parse_neural_backend(arg_or_default(args, "neural-backend", "aoti"));
    config.neural_device = parse_neural_device(arg_or_default(args, "neural-device", "mps"));
    config.neural_puct_config.root_noise_epsilon = std::stod(
        arg_or_default(args, "root-noise-epsilon", "0.25"));
    config.neural_puct_config.root_dirichlet_total_concentration = std::stod(
        arg_or_default(args, "root-dirichlet-total-concentration", "10.0"));
    config.neural_puct_config.root_noise_reference_actions = std::stod(
        arg_or_default(args, "root-noise-reference-actions", std::to_string(kCellCount)));
    config.neural_puct_config.root_noise_action_power = std::stod(
        arg_or_default(args, "root-noise-action-power", "0.5"));
    config.neural_puct_config.root_noise_empty_power = std::stod(
        arg_or_default(args, "root-noise-empty-power", "1.0"));
    config.visit_temperature_min = std::stod(
        arg_or_default(args, "visit-temperature-min", "0.05"));
    config.profile_neural_batch = arg_or_default(args, "profile-neural-batch", "0") != "0";

    if (config.games <= 0) {
        throw std::runtime_error("games must be positive");
    }
    if (config.simulations <= 0) {
        throw std::runtime_error("simulations must be positive");
    }
    if (config.threads <= 0) {
        throw std::runtime_error("threads must be positive");
    }
    if (config.max_actions <= 0) {
        throw std::runtime_error("max-actions must be positive");
    }
    if (config.neural_batch_size <= 0) {
        throw std::runtime_error("neural-batch-size must be positive");
    }
    if (config.neural_batch_wait_ms < 0.0) {
        throw std::runtime_error("neural-batch-wait-ms must be non-negative");
    }
    if (config.neural_puct_config.root_noise_epsilon < 0.0
            || config.neural_puct_config.root_noise_epsilon > 1.0) {
        throw std::runtime_error("root-noise-epsilon must be between 0 and 1");
    }
    if (config.neural_puct_config.root_dirichlet_total_concentration <= 0.0) {
        throw std::runtime_error("root-dirichlet-total-concentration must be positive");
    }
    if (config.neural_puct_config.root_noise_reference_actions <= 0.0) {
        throw std::runtime_error("root-noise-reference-actions must be positive");
    }
    if (config.neural_puct_config.root_noise_action_power < 0.0) {
        throw std::runtime_error("root-noise-action-power must be non-negative");
    }
    if (config.neural_puct_config.root_noise_empty_power < 0.0) {
        throw std::runtime_error("root-noise-empty-power must be non-negative");
    }
    if (config.visit_temperature_min <= 0.0) {
        throw std::runtime_error("visit-temperature-min must be positive");
    }
    return config;
}

void print_neural_batch_stats(const NeuralBatchStats& stats, double wait_ms) {
    auto milliseconds = [](std::uint64_t nanoseconds) {
        return static_cast<double>(nanoseconds) / 1'000'000.0;
    };
    const double average_batch_size = stats.batches == 0
        ? 0.0
        : static_cast<double>(stats.batch_items) / static_cast<double>(stats.batches);
    const double full_batch_rate = stats.batches == 0
        ? 0.0
        : static_cast<double>(stats.full_batches) / static_cast<double>(stats.batches);
    const double deadline_rate = stats.fill_waits == 0
        ? 0.0
        : static_cast<double>(stats.deadline_batches) / static_cast<double>(stats.fill_waits);
    const double average_fill_wait_ms = stats.fill_waits == 0
        ? 0.0
        : milliseconds(stats.fill_wait_ns) / static_cast<double>(stats.fill_waits);
    const double average_model_ms = stats.batches == 0
        ? 0.0
        : milliseconds(stats.model_time_ns) / static_cast<double>(stats.batches);
    const double average_feature_ms = stats.batches == 0
        ? 0.0
        : milliseconds(stats.feature_time_ns) / static_cast<double>(stats.batches);
    const double average_input_ms = stats.batches == 0
        ? 0.0
        : milliseconds(stats.input_time_ns) / static_cast<double>(stats.batches);
    const double average_inference_ms = stats.batches == 0
        ? 0.0
        : milliseconds(stats.inference_time_ns) / static_cast<double>(stats.batches);
    const double average_output_ms = stats.batches == 0
        ? 0.0
        : milliseconds(stats.output_time_ns) / static_cast<double>(stats.batches);
    const double average_postprocess_ms = stats.batches == 0
        ? 0.0
        : milliseconds(stats.postprocess_time_ns) / static_cast<double>(stats.batches);
    const double average_request_latency_ms = stats.requests == 0
        ? 0.0
        : milliseconds(stats.request_latency_ns) / static_cast<double>(stats.requests);

    std::cerr << std::fixed << std::setprecision(3)
              << "neural_batch_profile "
              << "wait_ms=" << wait_ms
              << " requests=" << stats.requests
              << " batches=" << stats.batches
              << " avg_batch_size=" << average_batch_size
              << " full_batch_rate=" << full_batch_rate
              << " fill_waits=" << stats.fill_waits
              << " deadline_batches=" << stats.deadline_batches
              << " deadline_rate=" << deadline_rate
              << " total_fill_wait_ms=" << milliseconds(stats.fill_wait_ns)
              << " avg_fill_wait_ms=" << average_fill_wait_ms
              << " total_model_ms=" << milliseconds(stats.model_time_ns)
              << " avg_model_ms=" << average_model_ms
              << " total_feature_ms=" << milliseconds(stats.feature_time_ns)
              << " avg_feature_ms=" << average_feature_ms
              << " total_input_ms=" << milliseconds(stats.input_time_ns)
              << " avg_input_ms=" << average_input_ms
              << " total_inference_ms=" << milliseconds(stats.inference_time_ns)
              << " avg_inference_ms=" << average_inference_ms
              << " total_output_ms=" << milliseconds(stats.output_time_ns)
              << " avg_output_ms=" << average_output_ms
              << " total_postprocess_ms=" << milliseconds(stats.postprocess_time_ns)
              << " avg_postprocess_ms=" << average_postprocess_ms
              << " avg_request_latency_ms=" << average_request_latency_ms
              << "\n";
}

TerminalInfo terminal_info(const TrackedState& state) {
    return {
        state.winner(),
        state.group_sizes(kPlayerOne),
        state.group_sizes(kPlayerTwo),
        kCellCount - state.empty_count(),
        state.completed_turns,
    };
}

std::vector<double> policy_target_from_root(const PuctNode* root) {
    std::vector<double> target(kMaxActions, 0.0);
    int total_visits = 0;
    for (const auto& edge : root->children) {
        total_visits += edge.child->visits;
    }
    if (total_visits == 0) {
        throw std::runtime_error("teacher returned no visited actions");
    }
    for (const auto& edge : root->children) {
        target[edge.action] = static_cast<double>(edge.child->visits) / total_visits;
    }
    return target;
}

double visit_temperature(const TrackedState& state, const Config& config) {
    double empty_ratio = static_cast<double>(state.empty_count()) / static_cast<double>(kCellCount);
    return std::max(config.visit_temperature_min, empty_ratio);
}

int sample_action_from_root(const PuctNode* root, double temperature, std::mt19937_64& rng) {
    std::vector<int> actions;
    std::vector<double> weights;
    actions.reserve(root->children.size());
    weights.reserve(root->children.size());

    double max_log_weight = -std::numeric_limits<double>::infinity();
    for (const auto& edge : root->children) {
        actions.push_back(edge.action);
        double log_weight = std::log(static_cast<double>(edge.child->visits)) / temperature;
        weights.push_back(log_weight);
        max_log_weight = std::max(max_log_weight, log_weight);
    }
    for (double& weight : weights) {
        weight = std::exp(weight - max_log_weight);
    }
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return actions[distribution(rng)];
}

std::vector<Sample> play_game(
    const Config& config,
    int game_id,
    NeuralEvaluatorBase* neural_evaluator) {
    TrackedState state;
    std::mt19937_64 rng(config.seed + static_cast<std::uint64_t>(game_id));
    std::vector<Sample> samples;

    for (int ply = 0; ply < config.max_actions; ++ply) {
        if (state.is_terminal()) {
            TerminalInfo terminal = terminal_info(state);
            for (auto& sample : samples) {
                sample.terminal = terminal;
                sample.value_target = state.result_for(sample.state.current_player);
            }
            return samples;
        }

        std::uint64_t search_seed = rng();
        int action = 0;
        std::vector<double> policy_target;
        if (config.teacher == TeacherKind::NeuralPuct) {
            NeuralPuctMcts search(
                config.simulations,
                search_seed,
                *neural_evaluator,
                config.neural_puct_config);
            PuctNode* root = search.search(state);
            action = sample_action_from_root(root, visit_temperature(state, config), rng);
            policy_target = policy_target_from_root(root);
        } else {
            PuctMcts search(config.simulations, search_seed, config.puct_config);
            PuctNode* root = search.search(state);
            action = sample_action_from_root(root, visit_temperature(state, config), rng);
            policy_target = policy_target_from_root(root);
        }

        samples.push_back({
            state,
            policy_target,
            {},
            game_id,
            ply,
            search_seed,
            action,
            0,
        });
        state.apply_action(action);
    }

    throw std::runtime_error("self-play game exceeded max-actions");
}

std::vector<std::vector<Sample>> generate_games(const Config& config) {
    int worker_count = std::min(config.threads, config.games);
    std::vector<std::vector<Sample>> games(static_cast<std::size_t>(config.games));
    std::unique_ptr<BatchedNeuralEvaluator> neural_evaluator;
    if (config.teacher == TeacherKind::NeuralPuct) {
        if (config.neural_backend == NeuralBackend::Mlx) {
            neural_evaluator = std::make_unique<BatchedNeuralEvaluator>(
                make_mlx_neural_batch_model(config.model_path, config.neural_batch_size),
                config.neural_batch_size,
                config.neural_batch_wait_ms);
        } else {
            neural_evaluator = std::make_unique<BatchedNeuralEvaluator>(
                make_aoti_neural_batch_model(
                    config.model_path,
                    config.neural_batch_size,
                    config.neural_device),
                config.neural_batch_size,
                config.neural_batch_wait_ms);
        }
    }
    std::vector<std::thread> workers;
    std::mutex error_mutex;
    std::optional<std::string> first_error;
    workers.reserve(worker_count);

    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            try {
                for (int game_id = worker; game_id < config.games; game_id += worker_count) {
                    games[game_id] = play_game(config, game_id, neural_evaluator.get());
                }
            } catch (const std::exception& exc) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error.has_value()) {
                    first_error = exc.what();
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    if (first_error.has_value()) {
        throw std::runtime_error(first_error.value());
    }
    if (config.profile_neural_batch && neural_evaluator) {
        print_neural_batch_stats(
            neural_evaluator->batch_stats(),
            config.neural_batch_wait_ms);
    }
    return games;
}

void write_int_array(std::ostream& out, const std::array<Owner, kCellCount>& values) {
    out << "[";
    for (int index = 0; index < kCellCount; ++index) {
        if (index != 0) {
            out << ",";
        }
        out << static_cast<int>(values[index]);
    }
    out << "]";
}

void write_int_vector(std::ostream& out, const std::vector<int>& values) {
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
}

void write_selected(std::ostream& out, const SelectedCells& selected) {
    out << "[";
    for (std::size_t index = 0; index < selected.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        out << static_cast<int>(selected[index]);
    }
    out << "]";
}

void write_legal_mask(std::ostream& out, const TrackedState& state) {
    std::array<bool, kMaxActions> legal{};
    ActionList actions = state.legal_actions();
    for (std::size_t index = 0; index < actions.size(); ++index) {
        legal[actions[index]] = true;
    }
    out << "[";
    for (int action = 0; action < kMaxActions; ++action) {
        if (action != 0) {
            out << ",";
        }
        out << (legal[action] ? "true" : "false");
    }
    out << "]";
}

void write_double_array(std::ostream& out, const std::vector<double>& values) {
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
}

void write_terminal(std::ostream& out, const TerminalInfo& terminal) {
    out << "{";
    out << "\"winner\":" << terminal.winner;
    out << ",\"blue_group_sizes\":";
    write_int_vector(out, terminal.blue_group_sizes);
    out << ",\"white_group_sizes\":";
    write_int_vector(out, terminal.white_group_sizes);
    out << ",\"filled_cells\":" << terminal.filled_cells;
    out << ",\"completed_turns\":" << terminal.completed_turns;
    out << "}";
}

std::string teacher_label(const Config& config) {
    if (config.teacher == TeacherKind::NeuralPuct) {
        return "neural-puct:" + std::to_string(config.simulations)
            + ":model=" + config.model_path
            + ":batch=" + std::to_string(config.neural_batch_size)
            + ":backend=" + neural_backend_label(config.neural_backend)
            + ":device=" + neural_device_label(config.neural_device)
            + ":root_noise_epsilon=" + std::to_string(
                config.neural_puct_config.root_noise_epsilon)
            + ":root_dirichlet_total_concentration=" + std::to_string(
                config.neural_puct_config.root_dirichlet_total_concentration)
            + ":root_noise_reference_actions=" + std::to_string(
                config.neural_puct_config.root_noise_reference_actions)
            + ":root_noise_action_power=" + std::to_string(
                config.neural_puct_config.root_noise_action_power)
            + ":root_noise_empty_power=" + std::to_string(
                config.neural_puct_config.root_noise_empty_power)
            + ":visit_temperature=max("
            + std::to_string(config.visit_temperature_min)
            + ",empty_count/61)";
    }
    return "puct:" + std::to_string(config.simulations)
        + ":prior=" + puct_prior_mode_name(config.puct_config.prior)
        + ":rollout=" + puct_rollout_mode_name(config.puct_config.rollout)
        + ":visit_temperature=max("
        + std::to_string(config.visit_temperature_min)
        + ",empty_count/61)";
}

void write_sample(std::ostream& out, const Sample& sample, const std::string& teacher) {
    const TrackedState& state = sample.state;
    out << "{\"state\":{";
    out << "\"owners\":";
    write_int_array(out, state.owners);
    out << ",\"current_player\":" << static_cast<int>(state.current_player);
    out << ",\"selected_this_turn\":";
    write_selected(out, state.selected);
    out << ",\"claimed_this_turn\":" << state.selected.size();
    out << ",\"max_claims\":" << static_cast<int>(state.max_claims);
    out << ",\"turn_start_largest\":" << static_cast<int>(state.turn_start_largest);
    out << ",\"opening_turn\":" << (state.opening_turn ? "true" : "false");
    out << ",\"legal_mask\":";
    write_legal_mask(out, state);
    out << "},\"policy_target\":";
    write_double_array(out, sample.policy_target);
    out << ",\"value_target\":" << sample.value_target;
    out << ",\"terminal\":";
    write_terminal(out, sample.terminal);
    out << ",\"meta\":{";
    out << "\"action_count\":" << kMaxActions;
    out << ",\"finish_action\":" << static_cast<int>(kFinish);
    out << ",\"teacher\":\"" << teacher << "\"";
    out << ",\"game_id\":" << sample.game_id;
    out << ",\"ply\":" << sample.ply;
    out << ",\"seed\":" << sample.search_seed;
    out << ",\"chosen_action\":" << sample.chosen_action;
    out << "}}\n";
}

int write_jsonl(const Config& config, const std::vector<std::vector<Sample>>& games) {
    std::filesystem::path output_path(config.output_path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(config.output_path);
    if (!out) {
        throw std::runtime_error("could not open output file: " + config.output_path);
    }
    std::string teacher = teacher_label(config);
    int sample_count = 0;
    for (const auto& samples : games) {
        for (const auto& sample : samples) {
            write_sample(out, sample, teacher);
            ++sample_count;
        }
    }
    return sample_count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse_config(argc, argv);
        auto games = generate_games(config);
        int sample_count = write_jsonl(config, games);
        int worker_count = std::min(config.threads, config.games);
        std::cout << "wrote " << sample_count << " samples to " << config.output_path
                  << " using " << worker_count << " threads\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
