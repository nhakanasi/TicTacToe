"""Play TicTacToe/Gomoku with unified agent selection."""
import argparse
import sys

from strat.config import set_board
from strat.judger import Judger
from strat.players import (
    HumanCLI, MCTSAgent, MinimaxAgent,
    AlphaBetaAgent, HeuristicAgent, TDPolicy, QTableAgent,
    MCTSSharedAgent, NeuralMCTSAgent
)


# All available agents/players
AGENTS = {
    "human": HumanCLI,
    "mcts": MCTSAgent,
    "minimax": MinimaxAgent,
    "alphabeta": AlphaBetaAgent,
    "heuristic": HeuristicAgent,
    "td": TDPolicy,
    "qtable": QTableAgent,
    "mcts-shared": MCTSSharedAgent,
    "neural-mcts": NeuralMCTSAgent,
}


def parse_player_arg(player_str: str):
    """Parse player argument in format: 'player1-vs-player2'
    
    Examples:
        'human-vs-mcts' -> ('human', 'mcts')
        'mcts-vs-alphabeta' -> ('mcts', 'alphabeta')
    """
    parts = player_str.lower().split('-vs-')
    if len(parts) != 2:
        raise ValueError(f"Player format must be 'player1-vs-player2', got: {player_str}")
    
    player1, player2 = parts[0].strip(), parts[1].strip()
    
    if player1 not in AGENTS:
        raise ValueError(f"Unknown player1: '{player1}'. Available: {', '.join(AGENTS.keys())}")
    if player2 not in AGENTS:
        raise ValueError(f"Unknown player2: '{player2}'. Available: {', '.join(AGENTS.keys())}")
    
    return player1, player2


def main():
    ap = argparse.ArgumentParser(
        description="Play TicTacToe/Gomoku with flexible player selection",
        epilog="Examples:\n"
               "  python play.py --player human-vs-mcts --rows 3 --cols 3 --win 3\n"
               "  python play.py --player mcts-vs-alphabeta --rows 5 --cols 5 --win 4\n"
               "  python play.py --player human-vs-heuristic --rows 10 --cols 10 --win 5",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rows", type=int, default=3, help="Board rows")
    ap.add_argument("--cols", type=int, default=3, help="Board cols")
    ap.add_argument("--win", type=int, default=3, help="Winning sequence length")
    ap.add_argument("--player", type=str, default="human-vs-mcts",
                    help=f"Player matchup (format: player1-vs-player2). "
                         f"Available: {', '.join(AGENTS.keys())}")
    ap.add_argument("--td-model", type=str, help="Path to TD model file (required for TD player)")
    ap.add_argument("--qtable-model", type=str, help="Path to Q-table model file (for Q-table player)")
    ap.add_argument("--mcts-shared-model", type=str, help="Path to MCTS shared tree file (for MCTS-shared)")
    ap.add_argument("--neural-model", type=str, help="Path to neural MCTS model (for neural-mcts)")
    ap.add_argument("--mcts-iterations", type=int, default=2000, help="MCTS iterations per move")
    ap.add_argument("--mcts-exploration", type=float, default=1.4, help="MCTS exploration constant (C in UCB formula)")
    ap.add_argument("--neural-simulations", type=int, default=800, help="Neural MCTS simulations per move")
    args = ap.parse_args()

    # Set global board configuration
    set_board(args.rows, args.cols, args.win)

    # Parse player selection
    try:
        player1_name, player2_name = parse_player_arg(args.player)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize players
    try:
        # Handle special players that need model paths or custom parameters
        if player1_name == "td":
            if not args.td_model:
                print("Error: --td-model required when using TD player")
                sys.exit(1)
            player1 = TDPolicy(model_path=args.td_model)
        elif player1_name == "qtable":
            player1 = QTableAgent(model_path=args.qtable_model, epsilon=0.0)
        elif player1_name == "mcts-shared":
            player1 = MCTSSharedAgent(
                num_simulations=args.mcts_iterations,
                model_path=args.mcts_shared_model,
                trained_mode=bool(args.mcts_shared_model)
            )
        elif player1_name == "neural-mcts":
            if not args.neural_model:
                print("Warning: --neural-model recommended for neural-mcts player")
            player1 = NeuralMCTSAgent(
                model_path=args.neural_model,
                num_simulations=args.neural_simulations
            )
        elif player1_name == "mcts":
            player1 = MCTSAgent(iterations=args.mcts_iterations, exploration_constant=args.mcts_exploration)
        else:
            player1 = AGENTS[player1_name]()

        if player2_name == "td":
            if not args.td_model:
                print("Error: --td-model required when using TD player")
                sys.exit(1)
            player2 = TDPolicy(model_path=args.td_model)
        elif player2_name == "qtable":
            player2 = QTableAgent(model_path=args.qtable_model, epsilon=0.0)
        elif player2_name == "mcts-shared":
            player2 = MCTSSharedAgent(
                num_simulations=args.mcts_iterations,
                model_path=args.mcts_shared_model,
                trained_mode=bool(args.mcts_shared_model)
            )
        elif player2_name == "neural-mcts":
            if not args.neural_model:
                print("Warning: --neural-model recommended for neural-mcts player")
            player2 = NeuralMCTSAgent(
                model_path=args.neural_model,
                num_simulations=args.neural_simulations
            )
        elif player2_name == "mcts":
            player2 = MCTSAgent(iterations=args.mcts_iterations, exploration_constant=args.mcts_exploration)
        else:
            player2 = AGENTS[player2_name]()

    except Exception as e:
        print(f"Error initializing players: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Game: {player1_name} vs {player2_name} | Board: {args.rows}x{args.cols}, Win: {args.win}")
    print("=" * 60)
    
    result = Judger(player1, player2).play(print_state=True)
    
    print("\n" + "=" * 60)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
