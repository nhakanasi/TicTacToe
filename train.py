"""Training script for TicTacToe/Gomoku agents."""
import argparse
import sys
import os

from strat.config import set_board


def train_td_agent(rows: int, cols: int, win: int, episodes: int, output_path: str):
    """Train a TD (Temporal Difference) agent."""
    set_board(rows, cols, win)
    try:
        from strat.train_heuristic import train
        print(f"Training TD agent: {rows}x{cols}, win={win}, episodes={episodes}")
        print(f"Output: {output_path}")
        print("-" * 60)
        train(episodes=episodes, output_path=output_path)
        print("-" * 60)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"Error training TD agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_neural_mcts(rows: int, cols: int, win: int, iterations: int, 
                      games_per_iter: int, simulations: int, output_path: str):
    """Train a Neural Network MCTS agent (AlphaGo-style)."""
    set_board(rows, cols, win)
    try:
        from strat.alpha_go import train_neural_mcts as train_nn
        print(f"Training Neural MCTS: {rows}x{cols}, win={win}")
        print(f"Iterations: {iterations}, Games/iter: {games_per_iter}, Simulations: {simulations}")
        print(f"Output: {output_path}")
        print("-" * 60)
        train_nn(
            iterations=iterations,
            games_per_iteration=games_per_iter,
            num_simulations=simulations,
            output_path=output_path
        )
        print("-" * 60)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"Error training Neural MCTS: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Train TicTacToe/Gomoku agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples:\n"
               "  python train.py --agent td --rows 10 --cols 10 --win 5 --episodes 10000\n"
               "  python train.py --agent neural-mcts --rows 5 --cols 5 --win 4 --iterations 100\n",
    )
    
    # Agent selection
    ap.add_argument("--agent", type=str, default="td", 
                    choices=["td", "neural-mcts"],
                    help="Agent type to train")
    
    # Board configuration
    ap.add_argument("--rows", type=int, default=3, help="Board rows")
    ap.add_argument("--cols", type=int, default=3, help="Board columns")
    ap.add_argument("--win", type=int, default=3, help="Winning sequence length")
    
    # Training parameters - TD
    ap.add_argument("--episodes", type=int, default=10000, 
                    help="Episodes for TD training")
    
    # Training parameters - Neural MCTS
    ap.add_argument("--iterations", type=int, default=1000, 
                    help="Iterations for Neural MCTS")
    ap.add_argument("--games-per-iter", type=int, default=10,
                    help="Self-play games per iteration (Neural MCTS)")
    ap.add_argument("--simulations", type=int, default=500,
                    help="MCTS simulations per move (Neural MCTS)")
    
    # Output
    ap.add_argument("--output", type=str, 
                    help="Output path for trained model (auto-generated if not specified)")
    
    args = ap.parse_args()

    # Auto-generate output path if not specified
    if not args.output:
        args.output = f"policy/board_{args.rows}x{args.cols}/{args.agent}_win{args.win}.pt"

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("=" * 60)
    print(f"Training Agent: {args.agent}")
    print(f"Board: {args.rows}x{args.cols}, Win: {args.win}")
    print("=" * 60)

    if args.agent == "td":
        train_td_agent(args.rows, args.cols, args.win, args.episodes, args.output)
    elif args.agent == "neural-mcts":
        train_neural_mcts(
            args.rows, args.cols, args.win, 
            args.iterations, args.games_per_iter, args.simulations,
            args.output
        )


if __name__ == "__main__":
    main()
