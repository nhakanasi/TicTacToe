"""Evaluation script for TicTacToe/Gomoku agents.

Evaluates different agents by running multiple games and collecting statistics.
Trains models if they don't exist.
"""
import argparse
import os
import sys
import time
from typing import Dict, List, Tuple
from datetime import datetime

from strat.config import set_board
from strat.judger import Judger
from strat.encode import Result
from strat.players import (
    HeuristicAgent, MCTSAgent, TDPolicy, 
    NeuralMCTSAgent, MCTSSharedAgent, QTableAgent,
    AlphaBetaAgent
)

# Global log file
LOG_FILE = None


def log(message: str):
    """Print and log message to file."""
    print(message)
    if LOG_FILE:
        LOG_FILE.write(message + "\n")
        LOG_FILE.flush()  # Flush immediately so data is saved


class EvaluationResult:
    """Store evaluation results."""
    def __init__(self, agent1_name: str, agent2_name: str):
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.draws = 0
        self.total_games = 0
        self.total_time = 0.0
    
    def add_result(self, result: Result):
        """Add a game result."""
        self.total_games += 1
        if result == Result.X_Wins:
            self.agent1_wins += 1
        elif result == Result.O_Wins:
            self.agent2_wins += 1
        else:
            self.draws += 1
    
    def print_summary(self):
        """Print evaluation summary."""
        avg_time = self.total_time / self.total_games if self.total_games > 0 else 0
        log(f"\n{'='*60}")
        log(f"Evaluation: {self.agent1_name} (X) vs {self.agent2_name} (O)")
        log(f"{'='*60}")
        log(f"Total games: {self.total_games}")
        log(f"{self.agent1_name} wins: {self.agent1_wins} ({self.agent1_wins/self.total_games*100:.1f}%)")
        log(f"{self.agent2_name} wins: {self.agent2_wins} ({self.agent2_wins/self.total_games*100:.1f}%)")
        log(f"Draws: {self.draws} ({self.draws/self.total_games*100:.1f}%)")
        log(f"Total time: {self.total_time:.2f}s | Avg time per game: {avg_time:.2f}s")
        log(f"{'='*60}")


def train_qtable_if_needed(model_path: str, epochs: int = 100000) -> float:
    """Train Q-table if model doesn't exist. Returns training time in seconds."""
    if os.path.exists(model_path):
        return 0.0
    
    sys.path.append(os.path.join(os.path.dirname(__file__), 'strat'))
    
    start_time = time.time()
    try:
        import importlib.util
        qtable_path = os.path.join(os.path.dirname(__file__), 'strat', 'q table.py')
        spec = importlib.util.spec_from_file_location("qtable_train", qtable_path)
        qtable_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qtable_module)
        
        qtable_module.train(epochs, print_every_n=epochs + 1)
        return time.time() - start_time
    except Exception:
        return 0.0


def train_mcts_shared_if_needed(model_path: str, epochs: int = 1000) -> float:
    """Train MCTS shared tree if model doesn't exist. Returns training time in seconds."""
    if os.path.exists(model_path):
        return 0.0
    return 0.0


def train_td_if_needed(model_path: str, rows: int, cols: int, win: int, episodes: int = 50000) -> float:
    """Train TD agent if model doesn't exist. Returns training time in seconds."""
    if os.path.exists(model_path):
        return 0.0
    
    log(f"\n[TD Training] Starting training for {rows}x{cols} board (win={win}), {episodes} episodes...")
    
    set_board(rows, cols, win)
    
    # Force reload of modules that cached config values
    import importlib
    if 'strat.train_heuristic' in sys.modules:
        importlib.reload(sys.modules['strat.train_heuristic'])
    if 'strat.encode' in sys.modules:
        importlib.reload(sys.modules['strat.encode'])
    
    start_time = time.time()
    try:
        from strat.train_heuristic import TDTicTacToe, TDTrainer
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        agent_x = TDTicTacToe(learning_rate=0.1, discount_factor=0.99)
        agent_o = TDTicTacToe(learning_rate=0.1, discount_factor=0.99)
        trainer = TDTrainer(agent_x, agent_o)
        
        trainer.train(
            num_training_samples=episodes,
            epsilon_start=0.2,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            eval_interval=episodes + 1,
            verbose=False
        )
        
        agent_x.save(model_path)
        elapsed = time.time() - start_time
        log(f"[TD Training] Complete! Saved to {model_path} ({elapsed/60:.2f} min)")
        return elapsed
    except Exception as e:
        log(f"[TD Training] ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        return 0.0


def train_neural_mcts_if_needed(model_path: str, rows: int, cols: int, win: int, iterations: int = 5) -> float:
    """Train Neural MCTS if model doesn't exist. Returns training time in seconds."""
    if os.path.exists(model_path):
        return 0.0
    
    # CRITICAL: Set board BEFORE importing training module
    set_board(rows, cols, win)
    
    start_time = time.time()
    try:
        from train_gomoku_alphago import train_alphago
        
        checkpoint_dir = os.path.dirname(model_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Use smaller values for faster training
        # 5 iterations, 5 games each, 100 simulations per move
        train_alphago(
            rows=rows,
            cols=cols,
            win=win,
            start_iteration=1,
            num_iterations=iterations,
            games_per_iteration=5,
            num_simulations=100,  # Reduced from 500
            batch_size=32,
            training_steps=200,   # Reduced from 500
            checkpoint_dir=checkpoint_dir
        )
        return time.time() - start_time
    except Exception as e:
        log(f"[ERROR] AlphaZero training failed for {rows}x{cols}: {e}")
        import traceback
        log(traceback.format_exc())
        return 0.0


def evaluate_agents(agent1, agent2, agent1_name: str, agent2_name: str, 
                    num_games: int, verbose: bool = False) -> EvaluationResult:
    """Evaluate two agents by playing multiple games."""
    result_tracker = EvaluationResult(agent1_name, agent2_name)
    
    log(f"\nPlaying {num_games} games: {agent1_name} vs {agent2_name}")
    
    start_time = time.time()
    for game_num in range(num_games):
        if game_num % 2 == 0:
            judger = Judger(agent1, agent2)
            result = judger.play(print_state=verbose)
        else:
            judger = Judger(agent2, agent1)
            result = judger.play(print_state=verbose)
            if result == Result.X_Wins:
                result = Result.O_Wins
            elif result == Result.O_Wins:
                result = Result.X_Wins
        
        result_tracker.add_result(result)
        
        # Progress update every 10 games or for slow matchups every game
        if (game_num + 1) % max(1, num_games // 10) == 0 or num_games <= 10:
            elapsed = time.time() - start_time
            log(f"  Game {game_num + 1}/{num_games} done ({elapsed:.1f}s)")
    
    result_tracker.total_time = time.time() - start_time
    return result_tracker


def evaluate_3x3(num_games: int = 100, verbose: bool = False):
    """Evaluate agents on 3x3 board against AlphaBeta pruning."""
    set_board(3, 3, 3)
    
    # Check/train models and track training times
    qtable_path = "policy_second.bin"
    mcts_shared_path = os.path.join("policy", "board_3x3", "mcts_tree.bin")
    td_path = os.path.join("policy", "board_3x3", "td_win3.pkl")
    
    training_times = {}
    training_times["Q-Table"] = train_qtable_if_needed(qtable_path, epochs=100000)
    training_times["MCTS-Shared"] = train_mcts_shared_if_needed(mcts_shared_path, epochs=1000)
    training_times["TD"] = train_td_if_needed(td_path, 3, 3, 3, episodes=50000)
    
    # Log training times
    trained_models = {k: v for k, v in training_times.items() if v > 0}
    if trained_models:
        log("\n" + "="*60)
        log("3x3 TRAINING TIMES")
        log("="*60)
        for model, t in trained_models.items():
            log(f"{model}: {t:.2f} seconds ({t/60:.2f} minutes)")
    
    # Initialize agents
    agents = {
        "MCTS": MCTSAgent(iterations=2000),
        "AlphaBeta": AlphaBetaAgent(depth=9),
        "Q-Table": QTableAgent(model_path=qtable_path, epsilon=0.0),
        "MCTS-Shared": MCTSSharedAgent(
            num_simulations=5000, 
            model_path=mcts_shared_path if os.path.exists(mcts_shared_path) else None,
            trained_mode=os.path.exists(mcts_shared_path)
        ),
        "TD": TDPolicy(model_path=td_path) if os.path.exists(td_path) else None,
        "Heuristic": HeuristicAgent(depth=10),
    }
    
    results = []
    
    matchups = [
        ("Q-Table", "AlphaBeta"),
        ("MCTS-Shared", "AlphaBeta"),
        ("MCTS", "AlphaBeta"),
        ("Heuristic", "AlphaBeta"),
        ("TD", "AlphaBeta") if agents["TD"] is not None else None,
    ]
    matchups = [m for m in matchups if m is not None]
    
    for agent1_name, agent2_name in matchups:
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        result = evaluate_agents(agent1, agent2, agent1_name, agent2_name, num_games, verbose)
        results.append(result)
    
    log("\n" + "="*60)
    log("3x3 EVALUATION RESULTS (vs AlphaBeta)")
    log("="*60)
    for result in results:
        result.print_summary()


def evaluate_5x5(num_games: int = 10, verbose: bool = False):
    """Evaluate agents on 5x5 board with win=5."""
    set_board(5, 5, 5)
    
    # Force reload of modules to pick up new config
    import importlib
    if 'strat.encode' in sys.modules:
        importlib.reload(sys.modules['strat.encode'])
    if 'strat.train_heuristic' in sys.modules:
        importlib.reload(sys.modules['strat.train_heuristic'])
    
    # Check/train models and track training times
    td_path = os.path.join("policy", "board_5x5", "td_win5.pkl")
    neural_mcts_path = os.path.join("policy", "board_5x5", "gomoku_final.pt")
    mcts_shared_path = os.path.join("policy", "board_5x5", "mcts_tree.bin")
    
    training_times = {}
    training_times["TD"] = train_td_if_needed(td_path, 5, 5, 5, episodes=50000)
    training_times["AlphaZero"] = train_neural_mcts_if_needed(neural_mcts_path, 5, 5, 5, iterations=5)
    training_times["MCTS-Shared"] = train_mcts_shared_if_needed(mcts_shared_path, epochs=1000)
    
    # Log training times
    trained_models = {k: v for k, v in training_times.items() if v > 0}
    if trained_models:
        log("\n" + "="*60)
        log("5x5 TRAINING TIMES")
        log("="*60)
        for model, t in trained_models.items():
            log(f"{model}: {t:.2f} seconds ({t/60:.2f} minutes)")
    
    # Initialize agents
    agents = {
        "Heuristic": HeuristicAgent(depth=4),
        "TD": TDPolicy(model_path=td_path) if os.path.exists(td_path) else None,
        "MCTS-Shared": MCTSSharedAgent(
            num_simulations=3000, 
            model_path=mcts_shared_path if os.path.exists(mcts_shared_path) else None,
            trained_mode=os.path.exists(mcts_shared_path)
        ),
        "AlphaZero": None,
    }
    
    # Try to load AlphaZero (may fail if model was trained for different board size)
    if os.path.exists(neural_mcts_path):
        try:
            agents["AlphaZero"] = NeuralMCTSAgent(model_path=neural_mcts_path, num_simulations=200)
        except Exception:
            pass
    
    results = []
    
    matchups = [
        ("TD", "Heuristic") if agents["TD"] is not None else None,
        ("MCTS-Shared", "Heuristic"),
        ("AlphaZero", "Heuristic") if agents["AlphaZero"] is not None else None,
    ]
    matchups = [m for m in matchups if m is not None]
    
    for agent1_name, agent2_name in matchups:
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        result = evaluate_agents(agent1, agent2, agent1_name, agent2_name, num_games, verbose)
        results.append(result)
    
    log("\n" + "="*60)
    log("5x5 (WIN=5) EVALUATION RESULTS")
    log("="*60)
    for result in results:
        result.print_summary()


def evaluate_10x10_gomoku(num_games: int = 5, verbose: bool = False):
    """Evaluate agents on 10x10 Gomoku (5-in-a-row)."""
    set_board(10, 10, 5)
    
    # Force reload of modules to pick up new config
    import importlib
    if 'strat.encode' in sys.modules:
        importlib.reload(sys.modules['strat.encode'])
    if 'strat.train_heuristic' in sys.modules:
        importlib.reload(sys.modules['strat.train_heuristic'])
    
    # Check/train models and track training times
    neural_path = os.path.join("policy", "board_10x10", "gomoku_final.pt")
    td_path = os.path.join("policy", "board_10x10", "td_win5.pkl")
    
    training_times = {}
    # Reduced from 100000 to 10000 for faster training (10x10 games are very slow)
    training_times["TD"] = train_td_if_needed(td_path, 10, 10, 5, episodes=10000)
    training_times["AlphaZero"] = train_neural_mcts_if_needed(neural_path, 10, 10, 5, iterations=5)
    
    # Log training times
    trained_models = {k: v for k, v in training_times.items() if v > 0}
    if trained_models:
        log("\n" + "="*60)
        log("10x10 TRAINING TIMES")
        log("="*60)
        for model, t in trained_models.items():
            log(f"{model}: {t:.2f} seconds ({t/60:.2f} minutes)")
    
    use_neural = os.path.exists(neural_path)
    
    # Initialize agents
    agents = {
        "Heuristic": HeuristicAgent(depth=3
                                    ),
        "TD": TDPolicy(model_path=td_path) if os.path.exists(td_path) else None,
    }
    
    if use_neural:
        try:
            agents["AlphaZero"] = NeuralMCTSAgent(model_path=neural_path, num_simulations=200)
        except Exception:
            use_neural = False
    
    results = []
    
    matchups = [
        ("TD", "Heuristic") if agents["TD"] is not None else None,
    ]
    
    if use_neural:
        matchups.append(("AlphaZero", "Heuristic"))
    
    matchups = [m for m in matchups if m is not None]
    
    for agent1_name, agent2_name in matchups:
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        result = evaluate_agents(agent1, agent2, agent1_name, agent2_name, num_games, verbose)
        results.append(result)
    
    log("\n" + "="*60)
    log("10x10 GOMOKU (WIN=5) EVALUATION RESULTS")
    log("="*60)
    for result in results:
        result.print_summary()


def main():
    global LOG_FILE
    
    parser = argparse.ArgumentParser(
        description="Evaluate TicTacToe/Gomoku agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--board", type=str, default="all",
                        choices=["all", "3x3", "5x5", "10x10"],
                        help="Which board size to evaluate")
    parser.add_argument("--games", type=int, default=None,
                        help="Number of games per matchup (auto if not specified)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each game state")
    parser.add_argument("--quick", action="store_true",
                        help="Quick evaluation with fewer games")
    
    args = parser.parse_args()
    
    # Open log file
    LOG_FILE = open("result.txt", "w", encoding="utf-8")
    
    start_time = datetime.now()
    
    log("="*60)
    log("TicTacToe/Gomoku Agent Evaluation")
    log("="*60)
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine number of games
    if args.games:
        games_3x3 = games_5x5 = games_10x10 = args.games
    elif args.quick:
        games_3x3 = 20
        games_5x5 = 5
        games_10x10 = 2
    else:
        games_3x3 = 100
        games_5x5 = 10
        games_10x10 = 5
    
    # Run evaluations
    if args.board in ["all", "3x3"]:
        evaluate_3x3(num_games=games_3x3, verbose=args.verbose)
    
    if args.board in ["all", "5x5"]:
        evaluate_5x5(num_games=games_5x5, verbose=args.verbose)
    
    if args.board in ["all", "10x10"]:
        evaluate_10x10_gomoku(num_games=games_10x10, verbose=args.verbose)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log("\n" + "="*60)
    log("EVALUATION COMPLETE")
    log("="*60)
    log(f"Total time: {duration/60:.2f} minutes")
    log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    LOG_FILE.close()


if __name__ == "__main__":
    main()
