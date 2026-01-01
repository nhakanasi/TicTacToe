"""
Temporal-Difference Learning for Tic Tac Toe using Linear Value Function
V(s) = W^T * X(s) where X(s) is a feature vector representing board state

Implementation follows the pseudocode from the paper:
- Collect training examples (feature vectors and target values)
- Apply LMS (Least Mean Squares) rule to update weights
- Use bootstrapping: V_train(b) = R + γ * V_hat(next_board)

Based on the approach in https://arxiv.org/pdf/2212.12252
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
try:
    from .encode import Board, Cell, Result, BOARD_ROWS, BOARD_COLS, WIN
except ImportError:
    from encode import Board, Cell, Result, BOARD_ROWS, BOARD_COLS, WIN


class TDTicTacToe:
    """
    Temporal-Difference learning agent for Tic Tac Toe.
    Value function: V(s) = W^T * X(s)
    
    Uses LMS rule for weight updates based on training examples collected from game trajectory.
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.99):
        """
        Initialize the TD learning agent.
        
        Args:
            learning_rate: Learning rate for LMS updates (alpha)
            discount_factor: Discount factor for future rewards (gamma)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.symbol = None
        
        # Initialize weights
        self.num_features = self._count_sequences()
        self.weights = np.zeros(self.num_features)
        
        # For tracking game history during training
        self.state_history = []
        self.feature_history = []
    
    @staticmethod
    def _count_sequences() -> int:
        """Count total number of sequences in board using existing logic."""
        count = 0
        # Horizontal sequences
        count += BOARD_ROWS * (BOARD_COLS - WIN + 1)
        # Vertical sequences
        count += BOARD_COLS * (BOARD_ROWS - WIN + 1)
        # Diagonal sequences
        count += (BOARD_ROWS - WIN + 1) * (BOARD_COLS - WIN + 1)
        # Anti-diagonal sequences
        count += (BOARD_ROWS - WIN + 1) * (BOARD_COLS - WIN + 1)
        return count
    
    def extract_features(self, board: Board) -> np.ndarray:
        """
        Extract feature vector from board state using Board.get_rows_cols_and_diagonals().
        
        Features represent the "potential" of each sequence:
        - Positive features for X-favorable patterns (X present, O absent)
        - Negative features for O-favorable patterns (O present, X absent)
        - Zero for blocked/empty patterns
        
        Args:
            board: Current board state
            
        Returns:
            Feature vector of shape (num_sequences,)
        """
        sequences = board.get_rows_cols_and_diagonals()
        features = np.zeros(len(sequences))
        
        for i, seq_sum in enumerate(sequences):
            # Sequence sum ranges from -WIN to +WIN
            # Positive sum: X-favorable (contains X, no O)
            # Negative sum: O-favorable (contains O, no X)
            # Zero: empty or blocked sequence
            
            if seq_sum > 0:
                # X-favorable: scale by number of X's
                features[i] = seq_sum / WIN  # Normalize to [0, 1)
            elif seq_sum < 0:
                # O-favorable: scale by number of O's
                features[i] = seq_sum / WIN  # Normalize to [-1, 0)
            else:
                # Empty sequence
                features[i] = 0
        
        return features
    
    def value(self, board: Board) -> float:
        """
        Compute value of a board state using V(s) = W^T * X(s).
        
        Args:
            board: Board state
            
        Returns:
            Estimated value (scalar)
        """
        features = self.extract_features(board)
        return np.dot(self.weights, features)
    
    def set_symbol(self, symbol: int):
        """Set the player symbol (Cell.X or Cell.O)."""
        self.symbol = symbol
    
    def reset(self):
        """Reset game history for new game."""
        self.state_history = []
        self.feature_history = []
    
    def act(self, board: Board) -> Optional[int]:
        """
        Choose action using greedy policy: best(legalmoves(b)).
        Selects move that maximizes V_hat(b) for current weights.
        
        Args:
            board: Current board state
            
        Returns:
            Action (1D index of move), or None if no valid moves
        """
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Greedy action selection: choose move that maximizes value
        best_move = None
        best_value = -np.inf
        
        for move in valid_moves:
            next_board = board.act(move, self.symbol)
            move_value = self.value(next_board)
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move
    
    def act_with_exploration(self, board: Board, epsilon: float = 0.1) -> Optional[int]:
        """
        Choose action with epsilon-greedy exploration.
        
        Args:
            board: Current board state
            epsilon: Probability of exploring (random action)
            
        Returns:
            Action (1D index of move), or None if no valid moves
        """
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Epsilon-greedy: explore with probability epsilon
        if np.random.random() < epsilon:
            return np.random.choice(valid_moves)
        
        # Otherwise use greedy policy
        return self.act(board)
    
    def record_state(self, board: Board):
        """Record board state and features for later training."""
        features = self.extract_features(board)
        self.state_history.append(board)
        self.feature_history.append(features)
    
    def update_from_game(self, final_reward: float):
        """
        Update weights using LMS (Least Mean Squares) rule following the paper's pseudocode.
        
        Algorithm:
        1. Collect all training examples: (featureV_i, V_train_i)
        2. For intermediate states: V_train(b_i) = 0 + γ * V_hat(b_{i+1})
        3. For final state: V_train(b_final) = final_reward
        4. Apply LMS update rule to all examples:
           W ← W + α * (V_train(b) - V_hat(b)) * featureV
        
        Args:
            final_reward: Final game result (+1 win, 0 draw, -1 loss)
                         Already adjusted for agent's perspective
        """
        if not self.feature_history:
            return
        
        # Step 1-2: Build training examples with bootstrapped targets
        training_examples = []
        
        # Process intermediate states (all but last)
        for t in range(len(self.feature_history) - 1):
            features_t = self.feature_history[t]
            
            # V_train(b_t) = r + γ * V_hat(b_{t+1})
            # where r = 0 for intermediate steps
            v_next = np.dot(self.weights, self.feature_history[t + 1])
            v_train = 0.0 + self.discount_factor * v_next
            
            training_examples.append((features_t, v_train))
        
        # Process final state
        if len(self.feature_history) > 0:
            features_final = self.feature_history[-1]
            # V_train(b_final) = final_reward (no next state)
            v_train_final = final_reward
            training_examples.append((features_final, v_train_final))
        
        # Step 3: Apply LMS rule to all training examples
        # For each training example: W ← W + α * (V_train(b) - V_hat(b)) * featureV
        for features, v_train in training_examples:
            # Calculate current estimate V_hat(b)
            v_hat = np.dot(self.weights, features)
            
            # Calculate error
            error = v_train - v_hat
            
            # Update rule (LMS): W ← W + α * error * featureV
            self.weights += self.learning_rate * error * features
    
    def save(self, path: str):
        """Save learned weights to file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)
    
    def load(self, path: str):
        """Load learned weights from file."""
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
    
    def get_weights(self) -> np.ndarray:
        """Get copy of current weights."""
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray):
        """Set weights directly."""
        self.weights = weights.copy()


class TDTrainer:
    """
    Trainer implementing the paper's training pseudocode.
    
    Input: numTrainingSamples(n)
    Output: targetweightVector(W), numAgentWins(nW), numAgentLoss(nL), numAgentDraws(nD)
    """
    
    def __init__(self, agent_x: TDTicTacToe, agent_o: TDTicTacToe):
        """
        Initialize trainer with two agents.
        
        Args:
            agent_x: TD learning agent for X
            agent_o: TD learning agent for O
        """
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.agent_x.set_symbol(Cell.X)
        self.agent_o.set_symbol(Cell.O)
        
        # Output statistics
        self.num_wins = 0      # nW: agent X wins
        self.num_losses = 0    # nL: agent X losses
        self.num_draws = 0     # nD: agent X draws
    
    def play_game(self, epsilon: float = 0.1) -> Result:
        board = Board()
        self.agent_x.reset()
        self.agent_o.reset()
        
        agents = [self.agent_x, self.agent_o]
        current_idx = 0
        
        # Play game until terminal state
        while True:
            current_agent = agents[current_idx]
            
            # Line 5: featureV = extractFeatures(board)
            current_agent.record_state(board)
            
            # Line 3: Choose best(legalmoves(b)) using current W
            if epsilon > 0:
                move = current_agent.act_with_exploration(board, epsilon)
            else:
                move = current_agent.act(board)
            
            if move is None:
                break
            
            # Execute move
            board = board.act(move, current_agent.symbol)
            
            # Check game end
            result = board.isEnd()
            if result != Result.Incomplete:
                break
            
            # Switch player
            current_idx = 1 - current_idx
        
        # Line 8: Calculate utility value of final board state
        # Line 11: Increment game status counts
        if result == Result.X_Wins:
            x_reward = 1.0
            o_reward = -1.0
            self.num_wins += 1
        elif result == Result.O_Wins:
            x_reward = -1.0
            o_reward = 1.0
            self.num_losses += 1
        else:  # Draw
            x_reward = 0.0
            o_reward = 0.0
            self.num_draws += 1
        
        # Line 12-18: Update W for all training examples using LMS rule
        self.agent_x.update_from_game(x_reward)
        self.agent_o.update_from_game(o_reward)
        
        return result
    
    def train(self, num_training_samples: int = 1000, epsilon_start: float = 0.2,
              epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
              eval_interval: int = 100):
        """
        Train both agents following paper's pseudocode with exploration decay.
        
        Input: numTrainingSamples(n)
        Output: targetweightVector(W), numAgentWins(nW), numAgentLoss(nL), numAgentDraws(nD)
        
        Initialisation: W=[0.5,0.5,0.5,0.5,0.5,0.5,0.5], trainGamesCount = 0,
        nW = nL = nD = 0
        
        Line 1: while trainGamesCount != n do
        Line 2-19: Play game and update weights with decaying exploration
        Line 20: return W, nW, nL, nD
        
        Args:
            num_training_samples: n - total number of training games
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate (exploration floor)
            epsilon_decay: Multiplicative decay factor per game (e.g., 0.995 = 0.5% decay)
            eval_interval: Interval for evaluation printout
        """
        print("=" * 70)
        print("TD Learning Training (Paper Pseudocode Implementation)")
        print("=" * 70)
        print(f"Target training games: {num_training_samples}")
        print(f"Exploration schedule:")
        print(f"  - Start epsilon: {epsilon_start}")
        print(f"  - End epsilon:   {epsilon_end}")
        print(f"  - Decay factor:  {epsilon_decay} (per game)")
        print(f"Learning rate (alpha): {self.agent_x.learning_rate}")
        print(f"Discount factor (gamma): {self.agent_x.discount_factor}")
        print("=" * 70)
        print()
        
        # Initialize epsilon for exploration decay
        epsilon = epsilon_start
        
        # Line 1: while trainGamesCount != n do
        for game_num in range(num_training_samples):
            # Line 2-19: Play game and update with current epsilon
            self.play_game(epsilon=epsilon)
            
            # Decay epsilon after each game: epsilon ← max(epsilon_end, epsilon * decay)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # Periodic evaluation
            if (game_num + 1) % eval_interval == 0:
                print(f"Games completed: {game_num + 1:5d}/{num_training_samples}")
                print(f"  Agent X - Wins: {self.num_wins:4d}, Losses: {self.num_losses:4d}, Draws: {self.num_draws:4d}")
                print(f"  Current epsilon: {epsilon:.6f}")
                print(f"  Weight vector W - shape: {self.agent_x.weights.shape}")
                print()
        
        print("=" * 70)
        print("Training Complete!")
        print(f"Final Results:")
        print(f"  Agent X Wins:   {self.num_wins}")
        print(f"  Agent X Losses: {self.num_losses}")
        print(f"  Agent X Draws:  {self.num_draws}")
        print(f"  Total games:    {self.num_wins + self.num_losses + self.num_draws}")
        print(f"  Final epsilon:  {epsilon:.6f}")
        print("=" * 70)
        
        # Line 20: return W, nW, nL, nD
        return (self.agent_x.weights, self.agent_o.weights,
                self.num_wins, self.num_losses, self.num_draws)

class TDPlayer:
    """
    Wrapper for playing games with a trained TD agent.
    Loads weights and provides utilities for inference.
    """
    
    def __init__(self, weights_path: str, symbol: int, learning_rate=0.1, discount_factor=0.99):
        """
        Initialize a player with trained weights.
        
        Args:
            weights_path: Path to saved weights file (.pkl)
            symbol: Player symbol (Cell.X or Cell.O)
            learning_rate: Learning rate (for compatibility, not used in inference)
            discount_factor: Discount factor (for compatibility, not used in inference)
        """
        self.agent = TDTicTacToe(learning_rate=learning_rate, discount_factor=discount_factor)
        self.agent.set_symbol(symbol)
        self.agent.load(weights_path)
        self.symbol = symbol
    
    def get_move(self, board: Board) -> Optional[int]:
        return self.agent.act(board)
    
    def get_move_with_value(self, board: Board) -> Tuple[Optional[int], float]:
        """
        Get best move and its value estimate.
        
        Args:
            board: Current board state
            
        Returns:
            Tuple of (move_index, estimated_value)
        """
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None, 0.0
        
        best_move = None
        best_value = -np.inf
        
        for move in valid_moves:
            next_board = board.act(move, self.symbol)
            move_value = self.agent.value(next_board)
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move, best_value



# Example usage and testing
if __name__ == "__main__":
    # Initialize agents and trainer
    agent_x = TDTicTacToe(learning_rate=0.1, discount_factor=0.99)
    agent_o = TDTicTacToe(learning_rate=0.1, discount_factor=0.99)
    trainer = TDTrainer(agent_x, agent_o)
    
    # Train agents following paper pseudocode with exploration decay
    w_x, w_o, nw, nl, nd = trainer.train(
        num_training_samples=100000,
        epsilon_start=0.2,      # Start with 20% exploration
        epsilon_end=0.01,       # End with 1% exploration (near-greedy)
        epsilon_decay=0.995,    # Decay by 0.5% each game
        eval_interval=100
    )
    
    # Save learned weights
    agent_x.save(rf"Reinforecedment learning\tictactoe\policy\td_agent_x.pkl_{BOARD_ROWS}_{BOARD_COLS}_{WIN}")
    agent_o.save(rf"Reinforecedment learning\tictactoe\policy\td_agent_o.pkl_{BOARD_ROWS}_{BOARD_COLS}_{WIN}")
    
    print(f"\nWeights saved to: policy/td_agent_x.pkl and model/td_agent_o.pkl_{BOARD_ROWS}_{BOARD_COLS}_{WIN}")