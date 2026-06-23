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
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

enum class Engine {
    Mcts,
    Puct,
    NeuralPuct,
};

struct AgentSpec {
    Engine engine = Engine::Mcts;
    int simulations = 1000;
    PuctConfig puct_config;
    std::string model_path;
    std::string label;
};

struct GameRecord {
    int pair_index = 0;
    int game_index = 0;
    std::string blue_agent;
    std::string white_agent;
    std::string blue_side;
    std::string white_side;
    std::string winner_side;
    std::string winner_agent;
    std::optional<int> winner_player;
    int completed_turns = 0;
    int internal_actions = 0;
    int filled_cells = 0;
};

struct Summary {
    int games = 0;
    int agent_a_wins = 0;
    int agent_b_wins = 0;
    int ties = 0;
    double agent_a_score_rate = 0.0;
    double ci_low = 0.0;
    double ci_high = 0.0;
    int a_blue_games = 0;
    int a_blue_wins = 0;
    int a_blue_losses = 0;
    int a_blue_ties = 0;
    int a_white_games = 0;
    int a_white_wins = 0;
    int a_white_losses = 0;
    int a_white_ties = 0;
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
    if (parts.size() < 2) {
        throw std::runtime_error(
            "agent must be mcts:N, puct:N:prior=flat|heuristic:rollout=flat|biased, "
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
    } else {
        throw std::runtime_error(
            "agent must be mcts:N, puct:N:prior=flat|heuristic:rollout=flat|biased, "
            "or neural-puct:N:MODEL.pt2");
    }
    return spec;
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

int choose_action(
    const AgentSpec& spec,
    const TrackedState& state,
    std::mt19937_64& rng,
    NeuralEvaluator* neural_evaluator) {
    if (spec.engine == Engine::Mcts) {
        Mcts search(spec.simulations, rng());
        return best_action(search.search(state));
    }

    if (spec.engine == Engine::Puct) {
        PuctMcts search(spec.simulations, rng(), spec.puct_config);
        return best_action(search.search(state));
    }

    NeuralPuctMcts search(spec.simulations, rng(), *neural_evaluator);
    return best_action(search.search(state));
}

GameRecord play_game(
    const AgentSpec& blue,
    const AgentSpec& white,
    const std::string& blue_side,
    const std::string& white_side,
    std::uint64_t seed,
    int pair_index,
    int game_index,
    int max_actions) {
    TrackedState state;
    std::mt19937_64 blue_rng(seed * 2 + 1);
    std::mt19937_64 white_rng(seed * 2 + 2);
    std::unique_ptr<NeuralEvaluator> blue_neural_evaluator;
    std::unique_ptr<NeuralEvaluator> white_neural_evaluator;
    if (blue.engine == Engine::NeuralPuct) {
        blue_neural_evaluator = std::make_unique<NeuralEvaluator>(blue.model_path);
    }
    if (white.engine == Engine::NeuralPuct) {
        white_neural_evaluator = std::make_unique<NeuralEvaluator>(white.model_path);
    }
    int internal_actions = 0;

    while (!state.is_terminal()) {
        if (internal_actions >= max_actions) {
            throw std::runtime_error("arena game exceeded max-actions");
        }
        if (state.current_player == kPlayerOne) {
            state.apply_action(
                choose_action(blue, state, blue_rng, blue_neural_evaluator.get()));
        } else {
            state.apply_action(
                choose_action(white, state, white_rng, white_neural_evaluator.get()));
        }
        ++internal_actions;
    }

    GameRecord record;
    record.pair_index = pair_index;
    record.game_index = game_index;
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
    } else if (record.winner_player == kPlayerTwo) {
        record.winner_side = white_side;
        record.winner_agent = white.label;
    }
    return record;
}

std::vector<GameRecord> run_arena(
    const AgentSpec& agent_a,
    const AgentSpec& agent_b,
    int pairs,
    std::uint64_t seed,
    int max_actions,
    int threads) {
    int worker_count = effective_thread_count(pairs, threads);
    std::vector<GameRecord> records(static_cast<std::size_t>(pairs) * 2);
    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            try {
                for (int pair_index = worker; pair_index < pairs; pair_index += worker_count) {
                    std::uint64_t first_seed =
                        seed + static_cast<std::uint64_t>(pair_index) * 2;
                    int first_record = pair_index * 2;
                    records[first_record] = play_game(
                        agent_a,
                        agent_b,
                        "A",
                        "B",
                        first_seed,
                        pair_index,
                        first_record,
                        max_actions);
                    records[first_record + 1] = play_game(
                        agent_b,
                        agent_a,
                        "B",
                        "A",
                        first_seed + 1,
                        pair_index,
                        first_record + 1,
                        max_actions);
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
        } else if (record.winner_side == "B") {
            ++summary.a_blue_losses;
        } else {
            ++summary.a_blue_ties;
        }
        return;
    }

    ++summary.a_white_games;
    if (record.winner_side == "A") {
        ++summary.a_white_wins;
    } else if (record.winner_side == "B") {
        ++summary.a_white_losses;
    } else {
        ++summary.a_white_ties;
    }
}

Summary summarize(const std::vector<GameRecord>& records) {
    Summary summary;
    summary.games = static_cast<int>(records.size());

    for (const auto& record : records) {
        if (record.winner_side == "A") {
            ++summary.agent_a_wins;
        } else if (record.winner_side == "B") {
            ++summary.agent_b_wins;
        } else {
            ++summary.ties;
        }
        add_color_result(record, summary);
        summary.average_completed_turns += record.completed_turns;
        summary.average_internal_actions += record.internal_actions;
        summary.average_filled_cells += record.filled_cells;
    }

    double a_score = summary.agent_a_wins + 0.5 * summary.ties;
    summary.agent_a_score_rate = a_score / summary.games;
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
    const Summary& summary) {
    std::cout << "Arena: A=" << agent_a.label << " vs B=" << agent_b.label << "\n";
    std::cout << "Pairs: " << pairs << "  Games: " << summary.games
              << "  Threads: " << threads << "  Seed: " << seed << "\n";
    std::cout << "Result: A wins " << summary.agent_a_wins
              << ", B wins " << summary.agent_b_wins
              << ", ties " << summary.ties << "\n";
    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);
    std::cout << "A score rate: " << summary.agent_a_score_rate * 100.0
              << "% (95% CI " << summary.ci_low * 100.0
              << "%.." << summary.ci_high * 100.0 << "%)\n";
    std::cout << "A as Blue: " << summary.a_blue_wins << "-"
              << summary.a_blue_losses << "-" << summary.a_blue_ties
              << " in " << summary.a_blue_games << " games\n";
    std::cout << "A as White: " << summary.a_white_wins << "-"
              << summary.a_white_losses << "-" << summary.a_white_ties
              << " in " << summary.a_white_games << " games\n";
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
    std::cout << ",\"summary\":{";
    std::cout << "\"games\":" << summary.games;
    std::cout << ",\"agent_a_wins\":" << summary.agent_a_wins;
    std::cout << ",\"agent_b_wins\":" << summary.agent_b_wins;
    std::cout << ",\"ties\":" << summary.ties;
    std::cout << ",\"agent_a_score_rate\":" << summary.agent_a_score_rate;
    std::cout << ",\"agent_a_score_ci95\":[" << summary.ci_low << "," << summary.ci_high << "]";
    std::cout << ",\"agent_a_as_blue\":{";
    std::cout << "\"games\":" << summary.a_blue_games;
    std::cout << ",\"wins\":" << summary.a_blue_wins;
    std::cout << ",\"losses\":" << summary.a_blue_losses;
    std::cout << ",\"ties\":" << summary.a_blue_ties << "}";
    std::cout << ",\"agent_a_as_white\":{";
    std::cout << "\"games\":" << summary.a_white_games;
    std::cout << ",\"wins\":" << summary.a_white_wins;
    std::cout << ",\"losses\":" << summary.a_white_losses;
    std::cout << ",\"ties\":" << summary.a_white_ties << "}";
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
        std::cout << ",\"blue_agent\":";
        print_json_string(record.blue_agent);
        std::cout << ",\"white_agent\":";
        print_json_string(record.white_agent);
        std::cout << ",\"blue_side\":";
        print_json_string(record.blue_side);
        std::cout << ",\"white_side\":";
        print_json_string(record.white_side);
        std::cout << ",\"winner_side\":";
        if (record.winner_side.empty()) {
            std::cout << "null";
        } else {
            print_json_string(record.winner_side);
        }
        std::cout << ",\"winner_agent\":";
        if (record.winner_agent.empty()) {
            std::cout << "null";
        } else {
            print_json_string(record.winner_agent);
        }
        std::cout << ",\"winner_player\":";
        if (record.winner_player.has_value()) {
            std::cout << record.winner_player.value();
        } else {
            std::cout << "null";
        }
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
        bool json = args.find("json") != args.end();

        int worker_count = effective_thread_count(pairs, threads);
        auto records = run_arena(agent_a, agent_b, pairs, seed, max_actions, threads);
        Summary summary = summarize(records);
        if (json) {
            print_json_report(agent_a, agent_b, pairs, worker_count, seed, summary, records);
        } else {
            print_text_report(agent_a, agent_b, pairs, worker_count, seed, summary);
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
