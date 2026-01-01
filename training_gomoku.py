import os
import sys
import numpy as np
import torch
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'strat'))

from strat.encode import Board, Result
from strat.alpha_go_large import AlphaGoTrainer

def main():
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    START_ITERATION = 2  # Resume from iteration 2
    NUM_ITERATIONS = 10  # Train for 10 more iterations
    GAMES_PER_ITERATION = 10
    NUM_SIMULATIONS = 500
    BATCH_SIZE = 32
    TRAINING_STEPS = 500
    CHECKPOINT_DIR = r'Reinforecedment learning\tictactoe\policy\gomoku_checkpoints'
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Training on {DEVICE}")
    print(f"Board size: {10}x{10}, Win condition: {5} in a row")
    print(f"Resume from iteration: {START_ITERATION}")
    print(f"Training for: {NUM_ITERATIONS} iterations")
    print(f"Simulations per move: {NUM_SIMULATIONS}, Batch size: {BATCH_SIZE}")
    print("-" * 80)
    
    # Load checkpoint from specific iteration
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'gomoku_iter_{START_ITERATION:02d}.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        trainer = AlphaGoTrainer(model_path=checkpoint_path, device=DEVICE)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Starting fresh training instead")
        trainer = AlphaGoTrainer(device=DEVICE)
        START_ITERATION = 1
    
    # Training loop
    for iteration in range(START_ITERATION, START_ITERATION + NUM_ITERATIONS):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}/{START_ITERATION + NUM_ITERATIONS - 1}")
        print(f"{'='*80}")
        
        iteration_start = datetime.now()
        
        # Self-play phase
        print(f"\nPhase 1: Self-play ({GAMES_PER_ITERATION} games)")
        print("-" * 80)
        
        games_played = 0
        x_wins = 0
        o_wins = 0
        draws = 0
        
        for game_idx in range(GAMES_PER_ITERATION):
            game_data, winner = trainer.play_game(num_simulations=NUM_SIMULATIONS)
            
            if "X wins" in str(winner):
                x_wins += 1
            elif "O wins" in str(winner):
                o_wins += 1
            else:
                draws += 1
            
            trainer.replay_buffer.extend(game_data)
            games_played += 1
            
            print(f"  Game {game_idx + 1}/{GAMES_PER_ITERATION}: {winner} "
                  f"({len(game_data)} moves) | Buffer size: {len(trainer.replay_buffer)}")
        
        print(f"\nGame results: X wins: {x_wins}, O wins: {o_wins}, Draws: {draws}")
        
        print(f"\nPhase 2: Network training ({TRAINING_STEPS} steps)")
        print("-" * 80)
        
        policy_losses = []
        value_losses = []
        
        for step in range(TRAINING_STEPS):
            policy_loss, value_loss = trainer.train_step(batch_size=BATCH_SIZE)
            
            if policy_loss is not None:
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                
                if (step + 1) % 50 == 0:
                    avg_policy_loss = np.mean(policy_losses[-50:])
                    avg_value_loss = np.mean(value_losses[-50:])
                    print(f"  Step {step + 1}/{TRAINING_STEPS}: "
                          f"Policy loss: {avg_policy_loss:.4f}, Value loss: {avg_value_loss:.4f}")
        
        if policy_losses:
            print(f"\nFinal losses: "
                  f"Policy: {np.mean(policy_losses[-100:]):.4f}, "
                  f"Value: {np.mean(value_losses[-100:]):.4f}")
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'gomoku_iter_{iteration:02d}.pt')
        torch.save(trainer.agent.model.state_dict(), checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
        
        iteration_time = (datetime.now() - iteration_start).total_seconds() / 60
        print(f"\nIteration time: {iteration_time:.2f} minutes")
    
    final_model_path = os.path.join(CHECKPOINT_DIR, 'gomoku_final.pt')
    torch.save(trainer.agent.model.state_dict(), final_model_path)
    print(f"\n{'='*80}")
    print(f"Training completed! Final model saved: {final_model_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()