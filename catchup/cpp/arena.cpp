#include "mcts.hpp"
#include "puct_mcts.hpp"
#include "puct_neural.hpp"
#include "tracked_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

enum class Engine {
    Random,
    Mcts,
    Puct,
    NeuralPuct,
};

enum class ActionSelection {
    MaxVisits,
    SampleVisits,
};

struct AgentSpec {
    Engine engine = Engine::Mcts;
    int simulations = 1000;
    PuctConfig puct_config;
    NeuralPuctConfig neural_puct_config;
    std::string model_path;
    std::string label;
};

struct ActionSelectionByAgent {
    ActionSelection agent_a = ActionSelection::MaxVisits;
    ActionSelection agent_b = ActionSelection::MaxVisits;
};

struct GameRecord {
    int pair_index = 0;
    int game_index = 0;
    std::uint64_t game_seed = 0;
    std::string blue_agent;
    std::string white_agent;
    std::string blue_side;
    std::string white_side;
    std::string winner_side;
    std::string winner_agent;
    int winner_player = kPlayerOne;
    int completed_turns = 0;
    int internal_actions = 0;
    int filled_cells = 0;
};

struct Summary {
    int games = 0;
    int agent_a_wins = 0;
    int agent_b_wins = 0;
    double agent_a_score_rate = 0.0;
    double ci_low = 0.0;
    double ci_high = 0.0;
    int a_blue_games = 0;
    int a_blue_wins = 0;
    int a_blue_losses = 0;
    int a_white_games = 0;
    int a_white_wins = 0;
    int a_white_losses = 0;
    int b_blue_games = 0;
    int b_blue_wins = 0;
    int b_blue_losses = 0;
    int b_white_games = 0;
    int b_white_wins = 0;
    int b_white_losses = 0;
    double average_completed_turns = 0.0;
    double average_internal_actions = 0.0;
    double average_filled_cells = 0.0;
};

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index) {
        std::string name = argv[index];
        if (name.rfind("--", 0) != 0) {
            throw std::runtime_error("unexpected positional argument: " + name);
        }
        std::string key = name.substr(2);
        if (key == "json") {
            args[key] = "1";
            continue;
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + name);
        }
        args[key] = argv[++index];
    }
    return args;
}

std::string arg_or_default(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& name,
    const std::string& default_value) {
    auto found = args.find(name);
    if (found == args.end()) {
        return default_value;
    }
    return found->second;
}

std::uint64_t mix_seed(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

std::uint64_t arena_game_seed(std::uint64_t base_seed, int pair_index, int game_in_pair) {
    std::uint64_t value = mix_seed(base_seed);
    value = mix_seed(value ^ static_cast<std::uint64_t>(pair_index));
    value = mix_seed(value ^ (static_cast<std::uint64_t>(game_in_pair) + 0xD1B54A32D192ED03ULL));
    return value;
}

std::vector<std::string> split_colon(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ':')) {
        parts.push_back(part);
    }
    return parts;
}

AgentSpec parse_agent_spec(const std::string& text) {
    auto parts = split_colon(text);
    if (parts.size() == 1 && parts[0] == "random") {
        AgentSpec spec;
        spec.engine = Engine::Random;
        spec.label = text;
        return spec;
    }

    if (parts.size() < 2) {
        throw std::runtime_error(
            "agent must be random, mcts:N, puct:N:prior=flat|heuristic:rollout=flat|biased, "
            "or neural-puct:N:MODEL.pt2");
    }

    std::string engine_name = parts[0];
    int simulations = std::stoi(parts[1]);
    if (simulations <= 0) {
        throw std::runtime_error("agent simulations must be positive");
    }

    AgentSpec spec;
    spec.simulations = simulations;
    spec.label = text;
    if (engine_name == "mcts") {
        if (parts.size() != 2) {
            throw std::runtime_error("mcts agent must be mcts:N");
        }
        spec.engine = Engine::Mcts;
    } else if (engine_name == "puct") {
        bool saw_prior = false;
        bool saw_rollout = false;
        spec.engine = Engine::Puct;
        for (std::size_t index = 2; index < parts.size(); ++index) {
            const std::string& option = parts[index];
            auto equals = option.find('=');
            if (equals == std::string::npos) {
                throw std::runtime_error(
                    "puct options must be prior=flat|heuristic or rollout=flat|biased");
            }
            std::string name = option.substr(0, equals);
            std::string value = option.substr(equals + 1);
            if (name == "prior") {
                spec.puct_config.prior = parse_puct_prior_mode(value);
                saw_prior = true;
            } else if (name == "rollout") {
                spec.puct_config.rollout = parse_puct_rollout_mode(value);
                saw_rollout = true;
            } else {
                throw std::runtime_error("unknown puct option: " + name);
            }
        }
        if (!saw_prior || !saw_rollout) {
            throw std::runtime_error(
                "puct agent must include prior=flat|heuristic and rollout=flat|biased");
        }
    } else if (engine_name == "neural-puct") {
        if (parts.size() < 3) {
            throw std::runtime_error("neural-puct agent must be neural-puct:N:MODEL.pt2");
        }
        spec.engine = Engine::NeuralPuct;
        spec.model_path = parts[2];
        for (std::size_t index = 3; index < parts.size(); ++index) {
            spec.model_path += ":" + parts[index];
        }
        spec.neural_puct_config.value_target =
            infer_neural_value_target_from_model_path(spec.model_path);
    } else {
        throw std::runtime_error(
            "agent must be random, mcts:N, puct:N:prior=flat|heuristic:rollout=flat|biased, "
            "or neural-puct:N:MODEL.pt2");
    }
    return spec;
}

ActionSelection parse_action_selection(const std::string& text) {
    if (text == "max") {
        return ActionSelection::MaxVisits;
    }
    if (text == "sample") {
        return ActionSelection::SampleVisits;
    }
    throw std::runtime_error("action selection must be max or sample");
}

std::string action_selection_label(ActionSelection action_selection) {
    return action_selection == ActionSelection::SampleVisits ? "sample" : "max";
}

std::string neural_value_target_label_for_agent(const AgentSpec& agent) {
    return agent.engine == Engine::NeuralPuct
        ? neural_value_target_label(agent.neural_puct_config.value_target)
        : "none";
}

bool parse_bool_arg(const std::string& text, const std::string& name) {
    if (text == "true" || text == "1") {
        return true;
    }
    if (text == "false" || text == "0") {
        return false;
    }
    throw std::runtime_error(name + " must be true or false");
}

bool uses_tanh_margin_value(const AgentSpec& agent) {
    return agent.engine == Engine::NeuralPuct
        && agent.neural_puct_config.value_target == NeuralValueTarget::TanhMarginScale6;
}

bool default_early_win_enabled(const AgentSpec& agent_a, const AgentSpec& agent_b) {
    return !uses_tanh_margin_value(agent_a) && !uses_tanh_margin_value(agent_b);
}

ActionSelectionByAgent default_action_selection(
    const AgentSpec& agent_a,
    const AgentSpec& agent_b) {
    ActionSelection default_selection =
        agent_a.engine == Engine::NeuralPuct && agent_b.engine == Engine::NeuralPuct
        ? ActionSelection::SampleVisits
        : ActionSelection::MaxVisits;
    return {default_selection, default_selection};
}

int effective_thread_count(int pairs, int threads) {
    if (pairs <= 0) {
        throw std::runtime_error("pairs must be positive");
    }
    if (threads <= 0) {
        throw std::runtime_error("threads must be positive");
    }
    return std::min(pairs, threads);
}

bool better_choice(int action, int visits, int best_action, int best_visits) {
    if (visits != best_visits) {
        return visits > best_visits;
    }
    return action < best_action;
}

int best_action(const Node* root) {
    auto iterator = root->children.begin();
    int best = static_cast<int>(iterator->first);
    int best_visits = iterator->second->visits;
    for (++iterator; iterator != root->children.end(); ++iterator) {
        const auto& child_entry = *iterator;
        int action = child_entry.first;
        int visits = child_entry.second->visits;
        if (better_choice(action, visits, best, best_visits)) {
            best = action;
            best_visits = visits;
        }
    }
    return best;
}

int best_action(const PuctNode* root) {
    const PuctChild& first = root->children.front();
    int best = first.action;
    int best_visits = first.child->visits;
    for (auto iterator = std::next(root->children.begin());
         iterator != root->children.end();
         ++iterator) {
        const auto& edge = *iterator;
        int action = edge.action;
        int visits = edge.child->visits;
        if (better_choice(action, visits, best, best_visits)) {
            best = action;
            best_visits = visits;
        }
    }
    return best;
}

int sample_action(const Node* root, std::mt19937_64& rng) {
    std::vector<int> actions;
    std::vector<int> weights;
    actions.reserve(root->children.size());
    weights.reserve(root->children.size());
    for (const auto& child_entry : root->children) {
        int visits = child_entry.second->visits;
        if (visits > 0) {
            actions.push_back(static_cast<int>(child_entry.first));
            weights.push_back(visits);
        }
    }
    if (actions.empty()) {
        return best_action(root);
    }
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return actions[distribution(rng)];
}

int sample_action(const PuctNode* root, std::mt19937_64& rng) {
    std::vector<int> actions;
    std::vector<int> weights;
    actions.reserve(root->children.size());
    weights.reserve(root->children.size());
    for (const auto& edge : root->children) {
        if (edge.child != nullptr && edge.child->visits > 0) {
            actions.push_back(edge.action);
            weights.push_back(edge.child->visits);
        }
    }
    if (actions.empty()) {
        return best_action(root);
    }
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return actions[distribution(rng)];
}

int choose_action(
    const AgentSpec& spec,
    const TrackedState& state,
    std::mt19937_64& rng,
    NeuralEvaluatorBase* neural_evaluator,
    ActionSelection action_selection) {
    if (spec.engine == Engine::Random) {
        ActionList actions = state.legal_actions();
        std::uniform_int_distribution<int> distribution(
            0,
            static_cast<int>(actions.size()) - 1);
        return actions[distribution(rng)];
    }

    if (spec.engine == Engine::Mcts) {
        Mcts search(spec.simulations, rng());
        const Node* root = search.search(state);
        return action_selection == ActionSelection::SampleVisits
            ? sample_action(root, rng)
            : best_action(root);
    }

    if (spec.engine == Engine::Puct) {
        PuctMcts search(spec.simulations, rng(), spec.puct_config);
        const PuctNode* root = search.search(state);
        return action_selection == ActionSelection::SampleVisits
            ? sample_action(root, rng)
            : best_action(root);
    }

    NeuralPuctMcts search(
        spec.simulations,
        rng(),
        *neural_evaluator,
        spec.neural_puct_config);
    const PuctNode* root = search.search(state);
    return action_selection == ActionSelection::SampleVisits
        ? sample_action(root, rng)
        : best_action(root);
}

GameRecord play_game(
    const AgentSpec& blue,
    const AgentSpec& white,
    NeuralEvaluatorBase* blue_neural_evaluator,
    NeuralEvaluatorBase* white_neural_evaluator,
    const std::string& blue_side,
    const std::string& white_side,
    std::uint64_t seed,
    int pair_index,
    int game_index,
    int max_actions,
    bool early_win_enabled,
    ActionSelection blue_action_selection,
    ActionSelection white_action_selection) {
    TrackedState state;
    state.early_win_enabled = early_win_enabled;
    std::mt19937_64 blue_rng(seed * 2 + 1);
    std::mt19937_64 white_rng(seed * 2 + 2);
    int internal_actions = 0;

    while (!state.is_terminal()) {
        if (internal_actions >= max_actions) {
            throw std::runtime_error("arena game exceeded max-actions");
        }
        if (state.current_player == kPlayerOne) {
            state.apply_action(
                choose_action(
                    blue,
                    state,
                    blue_rng,
                    blue_neural_evaluator,
                    blue_action_selection));
        } else {
            state.apply_action(
                choose_action(
                    white,
                    state,
                    white_rng,
                    white_neural_evaluator,
                    white_action_selection));
        }
        ++internal_actions;
    }

    GameRecord record;
    record.pair_index = pair_index;
    record.game_index = game_index;
    record.game_seed = seed;
    record.blue_agent = blue.label;
    record.white_agent = white.label;
    record.blue_side = blue_side;
    record.white_side = white_side;
    record.completed_turns = state.completed_turns;
    record.internal_actions = internal_actions;
    record.filled_cells = kCellCount - state.empty_count();

    record.winner_player = state.winner();
    if (record.winner_player == kPlayerOne) {
        record.winner_side = blue_side;
        record.winner_agent = blue.label;
    } else {
        record.winner_side = white_side;
        record.winner_agent = white.label;
    }
    return record;
}

struct ArenaNeuralEvaluators {
    std::unique_ptr<NeuralEvaluatorBase> agent_a_storage;
    std::unique_ptr<NeuralEvaluatorBase> agent_b_storage;
    NeuralEvaluatorBase* agent_a = nullptr;
    NeuralEvaluatorBase* agent_b = nullptr;
};

std::unique_ptr<BatchedNeuralEvaluator> make_batched_neural_evaluator(
    const std::string& model_path,
    int batch_size,
    double wait_ms,
    NeuralBackend backend,
    NeuralDevice device) {
    if (backend == NeuralBackend::Mlx) {
        return std::make_unique<BatchedNeuralEvaluator>(
            make_mlx_neural_batch_model(model_path, batch_size),
            batch_size,
            wait_ms);
    }
    return std::make_unique<BatchedNeuralEvaluator>(
        make_aoti_neural_batch_model(model_path, batch_size, device),
        batch_size,
        wait_ms);
}

ArenaNeuralEvaluators make_neural_evaluators(
    const AgentSpec& agent_a,
    const AgentSpec& agent_b,
    int batch_size,
    double wait_ms,
    NeuralBackend backend,
    NeuralDevice device) {
    ArenaNeuralEvaluators evaluators;
    if (agent_a.engine == Engine::NeuralPuct) {
        evaluators.agent_a_storage = make_batched_neural_evaluator(
            agent_a.model_path,
            batch_size,
            wait_ms,
            backend,
            device);
        evaluators.agent_a = evaluators.agent_a_storage.get();
    }
    if (agent_b.engine == Engine::NeuralPuct) {
        if (agent_a.engine == Engine::NeuralPuct && agent_a.model_path == agent_b.model_path) {
            evaluators.agent_b = evaluators.agent_a;
        } else {
            evaluators.agent_b_storage = make_batched_neural_evaluator(
                agent_b.model_path,
                batch_size,
                wait_ms,
                backend,
                device);
            evaluators.agent_b = evaluators.agent_b_storage.get();
        }
    }
    return evaluators;
}

std::vector<GameRecord> run_arena(
    const AgentSpec& agent_a,
    const AgentSpec& agent_b,
    int pairs,
    std::uint64_t seed,
    int max_actions,
    int threads,
    int neural_batch_size,
    double neural_batch_wait_ms,
    NeuralBackend neural_backend,
    NeuralDevice neural_device,
    bool early_win_enabled,
    ActionSelectionByAgent action_selection) {
    int worker_count = effective_thread_count(pairs, threads);
    std::vector<GameRecord> records(static_cast<std::size_t>(pairs) * 2);
    ArenaNeuralEvaluators neural_evaluators = make_neural_evaluators(
        agent_a,
        agent_b,
        neural_batch_size,
        neural_batch_wait_ms,
        neural_backend,
        neural_device);
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            try {
                for (int pair_index = worker; pair_index < pairs; pair_index += worker_count) {
                    std::uint64_t first_seed = arena_game_seed(seed, pair_index, 0);
                    std::uint64_t second_seed = arena_game_seed(seed, pair_index, 1);
                    int first_record = pair_index * 2;
                    records[first_record] = play_game(
                        agent_a,
                        agent_b,
                        neural_evaluators.agent_a,
                        neural_evaluators.agent_b,
                        "A",
                        "B",
                        first_seed,
                        pair_index,
                        first_record,
                        max_actions,
                        early_win_enabled,
                        action_selection.agent_a,
                        action_selection.agent_b);
                    records[first_record + 1] = play_game(
                        agent_b,
                        agent_a,
                        neural_evaluators.agent_b,
                        neural_evaluators.agent_a,
                        "B",
                        "A",
                        second_seed,
                        pair_index,
                        first_record + 1,
                        max_actions,
                        early_win_enabled,
                        action_selection.agent_b,
                        action_selection.agent_a);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return records;
}

void add_color_result(const GameRecord& record, Summary& summary) {
    if (record.blue_side == "A") {
        ++summary.a_blue_games;
        if (record.winner_side == "A") {
            ++summary.a_blue_wins;
        } else {
            ++summary.a_blue_losses;
        }
    } else {
        ++summary.a_white_games;
        if (record.winner_side == "A") {
            ++summary.a_white_wins;
        } else {
            ++summary.a_white_losses;
        }
    }

    if (record.blue_side == "B") {
        ++summary.b_blue_games;
        if (record.winner_side == "B") {
            ++summary.b_blue_wins;
        } else {
            ++summary.b_blue_losses;
        }
    } else {
        ++summary.b_white_games;
        if (record.winner_side == "B") {
            ++summary.b_white_wins;
        } else {
            ++summary.b_white_losses;
        }
    }
}

Summary summarize(const std::vector<GameRecord>& records) {
    Summary summary;
    summary.games = static_cast<int>(records.size());

    for (const auto& record : records) {
        if (record.winner_side == "A") {
            ++summary.agent_a_wins;
        } else {
            ++summary.agent_b_wins;
        }
        add_color_result(record, summary);
        summary.average_completed_turns += record.completed_turns;
        summary.average_internal_actions += record.internal_actions;
        summary.average_filled_cells += record.filled_cells;
    }

    summary.agent_a_score_rate = static_cast<double>(summary.agent_a_wins) / summary.games;
    double ci_radius = 1.96 * std::sqrt(
        summary.agent_a_score_rate * (1.0 - summary.agent_a_score_rate) / summary.games);
    summary.ci_low = std::max(0.0, summary.agent_a_score_rate - ci_radius);
    summary.ci_high = std::min(1.0, summary.agent_a_score_rate + ci_radius);
    summary.average_completed_turns /= summary.games;
    summary.average_internal_actions /= summary.games;
    summary.average_filled_cells /= summary.games;
    return summary;
}

void print_text_report(
    const AgentSpec& agent_a,
    const AgentSpec& agent_b,
    int pairs,
    int threads,
    std::uint64_t seed,
    NeuralBackend neural_backend,
    NeuralDevice neural_device,
    bool early_win_enabled,
    ActionSelectionByAgent action_selection,
    const Summary& summary) {
    std::cout << "Arena: A=" << agent_a.label << " vs B=" << agent_b.label << "\n";
    std::cout << "Pairs: " << pairs << "  Games: " << summary.games
              << "  Threads: " << threads << "  Seed: " << seed
              << "  Neural backend: " << neural_backend_label(neural_backend)
              << "  Neural device: " << neural_device_label(neural_device)
              << "  Early win: " << (early_win_enabled ? "true" : "false")
              << "  Action selection: A=" << action_selection_label(action_selection.agent_a)
              << " B=" << action_selection_label(action_selection.agent_b) << "\n";
    if (agent_a.engine == Engine::NeuralPuct || agent_b.engine == Engine::NeuralPuct) {
        std::cout << "Neural value target: A=" << neural_value_target_label_for_agent(agent_a)
                  << " B=" << neural_value_target_label_for_agent(agent_b) << "\n";
    }
    std::cout << "Result: A wins " << summary.agent_a_wins
              << ", B wins " << summary.agent_b_wins << "\n";
    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);
    std::cout << "A score rate: " << summary.agent_a_score_rate * 100.0
              << "% (95% CI " << summary.ci_low * 100.0
              << "%.." << summary.ci_high * 100.0 << "%)\n";
    std::cout << "A as Blue: " << summary.a_blue_wins << "-"
              << summary.a_blue_losses
              << " in " << summary.a_blue_games << " games\n";
    std::cout << "A as White: " << summary.a_white_wins << "-"
              << summary.a_white_losses
              << " in " << summary.a_white_games << " games\n";
    std::cout << "B as Blue: " << summary.b_blue_wins << "-"
              << summary.b_blue_losses
              << " in " << summary.b_blue_games << " games\n";
    std::cout << "B as White: " << summary.b_white_wins << "-"
              << summary.b_white_losses
              << " in " << summary.b_white_games << " games\n";
    std::cout << "Averages: " << summary.average_completed_turns << " turns, "
              << summary.average_internal_actions << " internal actions, "
              << summary.average_filled_cells << " filled cells\n";
}

void print_json_string(const std::string& text) {
    std::cout << "\"";
    for (char ch : text) {
        if (ch == '"' || ch == '\\') {
            std::cout << "\\" << ch;
        } else {
            std::cout << ch;
        }
    }
    std::cout << "\"";
}

void print_json_report(
    const AgentSpec& agent_a,
    const AgentSpec& agent_b,
    int pairs,
    int threads,
    std::uint64_t seed,
    NeuralBackend neural_backend,
    NeuralDevice neural_device,
    bool early_win_enabled,
    ActionSelectionByAgent action_selection,
    const Summary& summary,
    const std::vector<GameRecord>& records) {
    std::cout << "{";
    std::cout << "\"agent_a\":";
    print_json_string(agent_a.label);
    std::cout << ",\"agent_b\":";
    print_json_string(agent_b.label);
    std::cout << ",\"pairs\":" << pairs;
    std::cout << ",\"threads\":" << threads;
    std::cout << ",\"seed\":" << seed;
    std::cout << ",\"neural_backend\":";
    print_json_string(neural_backend_label(neural_backend));
    std::cout << ",\"neural_device\":";
    print_json_string(neural_device_label(neural_device));
    std::cout << ",\"early_win\":" << (early_win_enabled ? "true" : "false");
    std::cout << ",\"agent_a_action_selection\":";
    print_json_string(action_selection_label(action_selection.agent_a));
    std::cout << ",\"agent_b_action_selection\":";
    print_json_string(action_selection_label(action_selection.agent_b));
    std::cout << ",\"agent_a_neural_value_target\":";
    print_json_string(neural_value_target_label_for_agent(agent_a));
    std::cout << ",\"agent_b_neural_value_target\":";
    print_json_string(neural_value_target_label_for_agent(agent_b));
    std::cout << ",\"summary\":{";
    std::cout << "\"games\":" << summary.games;
    std::cout << ",\"agent_a_wins\":" << summary.agent_a_wins;
    std::cout << ",\"agent_b_wins\":" << summary.agent_b_wins;
    std::cout << ",\"agent_a_score_rate\":" << summary.agent_a_score_rate;
    std::cout << ",\"agent_a_score_ci95\":[" << summary.ci_low << "," << summary.ci_high << "]";
    std::cout << ",\"agent_a_as_blue\":{";
    std::cout << "\"games\":" << summary.a_blue_games;
    std::cout << ",\"wins\":" << summary.a_blue_wins;
    std::cout << ",\"losses\":" << summary.a_blue_losses << "}";
    std::cout << ",\"agent_a_as_white\":{";
    std::cout << "\"games\":" << summary.a_white_games;
    std::cout << ",\"wins\":" << summary.a_white_wins;
    std::cout << ",\"losses\":" << summary.a_white_losses << "}";
    std::cout << ",\"agent_b_as_blue\":{";
    std::cout << "\"games\":" << summary.b_blue_games;
    std::cout << ",\"wins\":" << summary.b_blue_wins;
    std::cout << ",\"losses\":" << summary.b_blue_losses << "}";
    std::cout << ",\"agent_b_as_white\":{";
    std::cout << "\"games\":" << summary.b_white_games;
    std::cout << ",\"wins\":" << summary.b_white_wins;
    std::cout << ",\"losses\":" << summary.b_white_losses << "}";
    std::cout << ",\"average_completed_turns\":" << summary.average_completed_turns;
    std::cout << ",\"average_internal_actions\":" << summary.average_internal_actions;
    std::cout << ",\"average_filled_cells\":" << summary.average_filled_cells;
    std::cout << "},\"games\":[";
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        if (index != 0) {
            std::cout << ",";
        }
        std::cout << "{";
        std::cout << "\"pair_index\":" << record.pair_index;
        std::cout << ",\"game_index\":" << record.game_index;
        std::cout << ",\"game_seed\":" << record.game_seed;
        std::cout << ",\"blue_agent\":";
        print_json_string(record.blue_agent);
        std::cout << ",\"white_agent\":";
        print_json_string(record.white_agent);
        std::cout << ",\"blue_side\":";
        print_json_string(record.blue_side);
        std::cout << ",\"white_side\":";
        print_json_string(record.white_side);
        std::cout << ",\"winner_side\":";
        print_json_string(record.winner_side);
        std::cout << ",\"winner_agent\":";
        print_json_string(record.winner_agent);
        std::cout << ",\"winner_player\":";
        std::cout << record.winner_player;
        std::cout << ",\"completed_turns\":" << record.completed_turns;
        std::cout << ",\"internal_actions\":" << record.internal_actions;
        std::cout << ",\"filled_cells\":" << record.filled_cells;
        std::cout << "}";
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        AgentSpec agent_a = parse_agent_spec(
            arg_or_default(args, "agent-a", "puct:1000:prior=heuristic:rollout=biased"));
        AgentSpec agent_b = parse_agent_spec(arg_or_default(args, "agent-b", "mcts:1000"));
        int pairs = std::stoi(arg_or_default(args, "pairs", "10"));
        std::uint64_t seed = static_cast<std::uint64_t>(
            std::stoull(arg_or_default(args, "seed", "1")));
        int max_actions = std::stoi(arg_or_default(args, "max-actions", "512"));
        int threads = std::stoi(arg_or_default(
            args,
            "threads",
            std::to_string(std::max(1u, std::thread::hardware_concurrency()))));
        int neural_batch_size = std::stoi(arg_or_default(args, "neural-batch-size", "32"));
        double neural_batch_wait_ms = std::stod(arg_or_default(args, "neural-batch-wait-ms", "0.5"));
        NeuralBackend neural_backend = parse_neural_backend(arg_or_default(args, "neural-backend", "aoti"));
        NeuralDevice neural_device = parse_neural_device(arg_or_default(args, "neural-device", "mps"));
        if (args.find("action-selection") != args.end()) {
            throw std::runtime_error(
                "use --agent-a-action-selection and --agent-b-action-selection");
        }
        ActionSelectionByAgent action_selection = default_action_selection(agent_a, agent_b);
        auto agent_a_action_selection_arg = args.find("agent-a-action-selection");
        if (agent_a_action_selection_arg != args.end()) {
            action_selection.agent_a =
                parse_action_selection(agent_a_action_selection_arg->second);
        }
        auto agent_b_action_selection_arg = args.find("agent-b-action-selection");
        if (agent_b_action_selection_arg != args.end()) {
            action_selection.agent_b =
                parse_action_selection(agent_b_action_selection_arg->second);
        }
        auto agent_a_neural_value_target_arg = args.find("agent-a-neural-value-target");
        if (agent_a_neural_value_target_arg != args.end()) {
            agent_a.neural_puct_config.value_target =
                parse_neural_value_target(agent_a_neural_value_target_arg->second);
        }
        auto agent_b_neural_value_target_arg = args.find("agent-b-neural-value-target");
        if (agent_b_neural_value_target_arg != args.end()) {
            agent_b.neural_puct_config.value_target =
                parse_neural_value_target(agent_b_neural_value_target_arg->second);
        }
        bool early_win_enabled = default_early_win_enabled(agent_a, agent_b);
        auto early_win_arg = args.find("early-win");
        if (early_win_arg != args.end()) {
            early_win_enabled = parse_bool_arg(early_win_arg->second, "early-win");
        }
        bool json = args.find("json") != args.end();

        int worker_count = effective_thread_count(pairs, threads);
        auto records = run_arena(
            agent_a,
            agent_b,
            pairs,
            seed,
            max_actions,
            threads,
            neural_batch_size,
            neural_batch_wait_ms,
            neural_backend,
            neural_device,
            early_win_enabled,
            action_selection);
        Summary summary = summarize(records);
        if (json) {
            print_json_report(
                agent_a,
                agent_b,
                pairs,
                worker_count,
                seed,
                neural_backend,
                neural_device,
                early_win_enabled,
                action_selection,
                summary,
                records);
        } else {
            print_text_report(
                agent_a,
                agent_b,
                pairs,
                worker_count,
                seed,
                neural_backend,
                neural_device,
                early_win_enabled,
                action_selection,
                summary);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
