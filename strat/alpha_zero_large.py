import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt, log
from collections import deque
import os
import sys

# Add paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from strat.encode import Board, Cell, Result
from strat import config  # Import module, not values, to get dynamic access
from model.upscalecnn import UpScaleCNNModel

C = 4.0 
DIRICHLET_ALPHA = 0.03
DIRICHLET_EPSILON = 0.25


class AlphaGoGomoku:
    class Node:
        def __init__(self, board: Board, parent=None, prior=0.0):
            self.board = board
            self.parent = parent
            self.prior = prior
            self.children = {}  # move -> Node
            
            self.visit_count = 0
            self.value_sum = 0.0
        
        def value(self):
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count
        
        def ucb_score(self, parent_visits):
            if self.visit_count == 0:
                return float('inf')
            
            exploitation = self.value()
            exploration = C * self.prior * sqrt(parent_visits) / (1 + self.visit_count)
            return exploitation + exploration
        
        def is_fully_expanded(self):
            valid_moves = self.board.get_valid_moves()
            return len(self.children) == len(valid_moves)
    
    def __init__(self, model_path=None, device='cpu', board_rows=None, board_cols=None):
        self.device = device
        # Use explicit dimensions if provided, otherwise fall back to config
        self.board_rows = board_rows if board_rows is not None else config.BOARD_ROWS
        self.board_cols = board_cols if board_cols is not None else config.BOARD_COLS
        self.board_size = self.board_rows * self.board_cols
        self.model = UpScaleCNNModel(board_rows=self.board_rows, board_cols=self.board_cols)
        self.model.to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.eval()
        
    def board_to_tensor(self, board: Board):

        board_2d = board.visualize()
        
        # Determine current player
        num_moves = np.count_nonzero(board.cells)
        current_player = Cell.X if num_moves % 2 == 0 else Cell.O
        opponent = Cell.O if current_player == Cell.X else Cell.X
        
        # Create planes - use board_2d.shape to ensure matching dimensions
        current_layer = (board_2d == current_player).astype(np.float32)
        opponent_layer = (board_2d == opponent).astype(np.float32)
        empty_layer = (board_2d == Cell.Empty).astype(np.float32)
        color_layer = np.ones_like(board_2d, dtype=np.float32) * (1 if current_player == Cell.X else 0)
        
        state = np.stack([current_layer, opponent_layer, empty_layer, color_layer], axis=0)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_policy_and_value(self, board: Board):
        state = self.board_to_tensor(board)
        
        with torch.no_grad():
            policy_logits, value = self.model(state)
        
        policy = policy_logits[0].cpu().numpy()
        value = value.item()
        
        # Mask invalid moves
        valid_moves = board.get_valid_moves()
        policy_masked = np.full(self.board_size, -np.inf)
        policy_masked[valid_moves] = policy[valid_moves]
        
        # Apply softmax
        policy_softmax = self._softmax(policy_masked)
        
        # Return only valid move probabilities
        policy_valid = np.zeros(self.board_size)
        policy_valid[valid_moves] = policy_softmax[valid_moves]
        
        return policy_valid, value
    
    @staticmethod
    def _softmax(x):
        """Numerically stable softmax"""
        # Handle case where all values are -inf
        max_x = np.max(x)
        if np.isinf(max_x) and max_x < 0:
            # All values are -inf, return uniform distribution over non-inf values
            # This shouldn't happen in practice, but prevents NaN
            result = np.zeros_like(x)
            valid = ~np.isinf(x)
            if np.any(valid):
                result[valid] = 1.0 / np.sum(valid)
            return result
        
        x = x - max_x
        e_x = np.exp(x)
        return e_x / e_x.sum()
    
    def select_child(self, node: Node):
        """Select child with highest UCB score"""
        best_child = None
        best_score = -np.inf
        
        for move, child in node.children.items():
            score = child.ucb_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand_node(self, node: Node):
        policy, _ = self.get_policy_and_value(node.board)
        valid_moves = node.board.get_valid_moves()
        
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(valid_moves))
        policy_noisy = np.zeros(config.BOARD_SIZE)
        policy_noisy[valid_moves] = (1 - DIRICHLET_EPSILON) * policy[valid_moves] + \
                                   DIRICHLET_EPSILON * noise
        policy = policy_noisy
        
        for move in valid_moves:
            if move not in node.children:
                new_board = node.board.act(move)
                child = self.Node(new_board, parent=node, prior=policy[move])
                node.children[move] = child

    def simulate(self, node: Node):
        path = [node]
        
        # Selection phase
        while not node.board.isEnd() == Result.Incomplete and node.children:
            node = self.select_child(node)
            path.append(node)
        
        # Expansion & Evaluation
        result = node.board.isEnd()
        
        if result == Result.Incomplete:
            self.expand_node(node)
            
            if node.children:
                move = np.random.choice(list(node.children.keys()))
                node = node.children[move]
                path.append(node)
        
        if result != Result.Incomplete:
            value = self._get_terminal_value(result)
        else:
            _, value = self.get_policy_and_value(node.board)
        
        self._backup(path, value)
    
    def _get_terminal_value(self, result):
        if result == Result.Draw:
            return 0.0
        elif result == Result.X_Wins:
            return 1.0  # X wins
        else:  # result == Result.O_Wins
            return -1.0  # O wins
    
    def _backup(self, path, value):
        for i, node in enumerate(reversed(path)):
            node.visit_count += 1
            node.value_sum += value if i % 2 == 0 else -value
    
    def search(self, board: Board, num_simulations=800, temperature=1.0):
        root = self.Node(board)
        self.expand_node(root)
        
        # Run simulations
        for _ in range(num_simulations):
            self.simulate(root)
        
        # Get move distribution
        valid_moves = board.get_valid_moves()
        
        if temperature == 0:
            # Deterministic: choose most visited
            best_move = max(valid_moves, 
                          key=lambda m: root.children[m].visit_count if m in root.children else 0)
            # Policy is one-hot for deterministic case
            policy = np.zeros(config.BOARD_SIZE)
            policy[best_move] = 1.0
        else:
            # Stochastic: sample according to visit counts
            visits = np.array([root.children[m].visit_count if m in root.children else 0 
                             for m in range(config.BOARD_SIZE)])
            visits_valid = visits.copy()
            visits_valid[~np.array([m in valid_moves for m in range(config.BOARD_SIZE)])] = 0
            
            # Temperature scaling
            if np.sum(visits_valid) > 0:
                probs = np.power(visits_valid, 1.0 / temperature)
                probs = probs / np.sum(probs)
                best_move = np.random.choice(config.BOARD_SIZE, p=probs)
            else:
                best_move = np.random.choice(valid_moves)
            
            # Return policy for training
            policy = np.zeros(config.BOARD_SIZE)
            for m in valid_moves:
                if m in root.children:
                    policy[m] = (root.children[m].visit_count) ** (1.0 / temperature)
            policy = policy / (np.sum(policy) + 1e-10)
        
        return best_move, policy, root


class AlphaGoTrainer:
    def __init__(self, model_path=None, device='cpu', board_rows=None, board_cols=None):
        self.agent = AlphaGoGomoku(model_path=model_path, device=device, 
                                   board_rows=board_rows, board_cols=board_cols)
        self.device = device
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=100000)
    
    def play_game(self, num_simulations=500, temperature_moves=30):
        board = Board()
        game_data = []
        
        move_count = 0
        while board.isEnd() == Result.Incomplete:
            move, policy, root = self.agent.search(board, 
                                                   num_simulations=num_simulations,
                                                   temperature=1.0 if move_count < temperature_moves else 0)
            
            # Store training data
            state = self.agent.board_to_tensor(board).cpu().numpy()
            game_data.append((state, policy, None))  # value filled later
            
            board = board.act(move)
            move_count += 1
        
        # Fill in terminal values
        result = board.isEnd()
        final_value = self.agent._get_terminal_value(result)
        
        for i in range(len(game_data)):
            state, policy, _ = game_data[i]
            # Alternate perspective
            value = final_value if i % 2 == 0 else -final_value
            game_data[i] = (state, policy, value)
        
        return game_data, board.winner
    
    def train_step(self, batch_size=32):
        """Train on batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample batch
        batch = [self.replay_buffer[np.random.randint(len(self.replay_buffer))] 
                for _ in range(batch_size)]
        
        states = np.concatenate([item[0] for item in batch])
        policies = np.array([item[1] for item in batch])
        values = np.array([item[2] for item in batch])
        
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        values_t = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        
        # Forward pass
        policy_logits, value_pred = self.agent.model(states_t)
        
        # Losses
        policy_loss = -torch.sum(policies_t * torch.log_softmax(policy_logits, dim=1)) / batch_size
        value_loss = nn.MSELoss()(value_pred, values_t)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()