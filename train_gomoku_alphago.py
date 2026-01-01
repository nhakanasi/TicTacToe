"""Training script for AlphaGo-style neural MCTS on Gomoku."""
import argparse
import os
import sys
from datetime import datetime

import torch
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'strat'))

from strat.config import set_board
from strat.encode import Board, Result
from strat.alpha_go_large import AlphaGoTrainer


def train_alphago(
    rows: int,
    cols: int,
    win: int,
    start_iteration: int = 1,
    num_iterations: int = 10,
    games_per_iteration: int = 10,
    num_simulations: int = 500,
    batch_size: int = 32,
    training_steps: int = 500,
    checkpoint_dir: str = "policy/gomoku_checkpoints",
    resume_from: str = None
):
    """Train AlphaGo-style neural MCTS agent.
    
    Args:
        rows: Board rows
        cols: Board columns
        win: Winning sequence length
        start_iteration: Starting iteration number (for resuming)
        num_iterations: Number of training iterations
        games_per_iteration: Self-play games per iteration
        num_simulations: MCTS simulations per move
        batch_size: Training batch size
        training_steps: Training steps per iteration
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Set board configuration
    set_board(rows, cols, win)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"AlphaGo Gomoku Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Board: {rows}x{cols}, Win: {win} in a row")
    print(f"Iterations: {start_iteration} to {start_iteration + num_iterations - 1}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Simulations per move: {num_simulations}")
    print(f"Training: {training_steps} steps, batch size {batch_size}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("-" * 80)
    
    # Initialize or load trainer
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint: {resume_from}")
        trainer = AlphaGoTrainer(model_path=resume_from, device=device)
    else:
        if resume_from:
            print(f"Warning: Checkpoint not found: {resume_from}")
        print("Starting fresh training")
        trainer = AlphaGoTrainer(device=device)
    
    # Training loop
    for iteration in range(start_iteration, start_iteration + num_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{start_iteration + num_iterations - 1}")
        print(f"{'='*80}")
        
        iteration_start = datetime.now()
        
        # Self-play phase
        print(f"\nPhase 1: Self-play ({games_per_iteration} games)")
        print("-" * 80)
        
        stats = {"X": 0, "O": 0, "Draw": 0}
        
        for game_idx in range(games_per_iteration):
            game_data, winner = trainer.play_game(num_simulations=num_simulations)
            
            # Track statistics
            winner_str = str(winner)
            if "X wins" in winner_str:
                stats["X"] += 1
            elif "O wins" in winner_str:
                stats["O"] += 1
            else:
                stats["Draw"] += 1
            
            trainer.replay_buffer.extend(game_data)
            
            print(f"  Game {game_idx + 1}/{games_per_iteration}: {winner} "
                  f"({len(game_data)} moves) | Buffer: {len(trainer.replay_buffer)}")
        
        print(f"\nGame results: X wins: {stats['X']}, O wins: {stats['O']}, "
              f"Draws: {stats['Draw']}")
        
        # Training phase
        print(f"\nPhase 2: Network training ({training_steps} steps)")
        print("-" * 80)
        
        policy_losses = []
        value_losses = []
        
        for step in range(training_steps):
            policy_loss, value_loss = trainer.train_step(batch_size=batch_size)
            
            if policy_loss is not None:
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
            
            # Progress update every 100 steps
            if (step + 1) % 100 == 0:
                if policy_losses:
                    avg_p = sum(policy_losses[-100:]) / len(policy_losses[-100:])
                    avg_v = sum(value_losses[-100:]) / len(value_losses[-100:])
                    print(f"  Step {step + 1}/{training_steps}: "
                          f"Policy Loss: {avg_p:.4f}, Value Loss: {avg_v:.4f}")
        
        # Summary
        if policy_losses:
            avg_policy_loss = sum(policy_losses) / len(policy_losses)
            avg_value_loss = sum(value_losses) / len(value_losses)
            print(f"\nTraining summary:")
            print(f"  Avg Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Avg Value Loss: {avg_value_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'gomoku_iter_{iteration:02d}.pt')
        torch.save(trainer.agent.model.state_dict(), checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
        
        iteration_time = (datetime.now() - iteration_start).total_seconds() / 60
        print(f"Iteration time: {iteration_time:.2f} minutes")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'gomoku_final.pt')
    torch.save(trainer.agent.model.state_dict(), final_model_path)
    print(f"\n{'='*80}")
    print(f"Training completed! Final model saved: {final_model_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Train AlphaGo-style neural MCTS for Gomoku",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Board configuration
    parser.add_argument("--rows", type=int, default=10, help="Board rows")
    parser.add_argument("--cols", type=int, default=10, help="Board columns")
    parser.add_argument("--win", type=int, default=5, help="Winning sequence length")
    
    # Training parameters
    parser.add_argument("--start-iter", type=int, default=1,
                        help="Starting iteration (for resuming)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=10,
                        help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=500,
                        help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--training-steps", type=int, default=500,
                        help="Training steps per iteration")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="policy/gomoku_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume-from", type=str,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train_alphago(
        rows=args.rows,
        cols=args.cols,
        win=args.win,
        start_iteration=args.start_iter,
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
