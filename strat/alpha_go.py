import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt
import os
import sys

# Add parent directory to path to access model module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from encode import Board, Cell, Result, BOARD_ROWS, BOARD_COLS, BOARD_SIZE
from model.simplecnn import SimpleCNN

C = 4  # UCB exploration constant

class NeuralMCTS:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.canonical_cache = {}  # Cache for canonical forms
        self.node_storage = {}     # Global storage for nodes (space saving)
    
    class Node:
        def __init__(self, board: Board, parent=None, parent_move=None, prior=0):
            self.original_board = board  # Keep original for move generation
            # Use Board's canonical hash method
            self.canonical_hash = board.get_canonical_hash()
            self.parent = parent
            self.parent_move = parent_move
            self.children = {}
            self.visits = 0
            self.value_sum = 0
            self.prior = prior
        
        def value(self):
            if self.visits == 0:
                return 0
            return self.value_sum / self.visits
        
        def ucb_score(self, parent_visits):
            if self.visits == 0:
                return float('inf')
            
            exploitation = self.value()
            exploration = C * self.prior * sqrt(parent_visits) / (1 + self.visits)
            return exploitation + exploration
        
        def is_expanded(self):
            return len(self.children) > 0

    def get_policy_value(self, board: Board):
        """Get policy and value from neural network with proper move mapping"""
        canonical_hash = board.get_canonical_hash()
        
        # Check cache first
        if canonical_hash in self.canonical_cache:
            return self.canonical_cache[canonical_hash]
        
        # Get canonical form and transformation info
        canonical_cells, transform_id = board.get_canonical()
        canonical_board = Board(canonical_cells)
        
        # Get network prediction for canonical board
        state = self.board_to_tensor(canonical_board)
        
        with torch.no_grad():
            policy, value = self.model(state)
            policy = policy.cpu().numpy()[0]
            value = value.cpu().item()
        
        # Transform policy back to original board orientation
        if transform_id != 0:
            # Convert policy to 2D for transformation
            policy_1d = np.zeros(BOARD_SIZE)
            for move in range(BOARD_SIZE):
                # Transform canonical move back to original board space
                original_move = Board.inverse_transform_move_1d(move, transform_id)
                policy_1d[original_move] = policy[move]
            transformed_policy = policy_1d
        else:
            transformed_policy = policy
        
        # Cache the result
        self.canonical_cache[canonical_hash] = (transformed_policy, value)
        
        return transformed_policy, value
    
    def board_to_tensor(self, board: Board):
        """Convert board to tensor from current player's perspective"""
        board_2d = board.visualize()
        current_player = board.next_player()
        opponent = Cell.O if current_player == Cell.X else Cell.X
        
        # Always: Channel 0 = current player, Channel 1 = opponent, Channel 2 = empty
        current_layer = (board_2d == current_player).astype(np.float32)
        opponent_layer = (board_2d == opponent).astype(np.float32)
        empty_layer = (board_2d == Cell.Empty).astype(np.float32)
        
        state = np.stack([current_layer, opponent_layer, empty_layer], axis=0)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def get_or_create_node(self, board: Board, parent=None, parent_move=None, prior=0):
        """Get existing node or create new one using canonical hash for storage"""
        canonical_hash = board.get_canonical_hash()
        
        if canonical_hash in self.node_storage:
            return self.node_storage[canonical_hash]
        else:
            # Create new node
            new_node = self.Node(board, parent, parent_move, prior)
            self.node_storage[canonical_hash] = new_node
            return new_node

    def search(self, board: Board, num_simulations=800, ai_symbol=None):
        """MCTS search"""
        if ai_symbol is None:
            ai_symbol = board.next_player()
        
        self.ai_symbol = ai_symbol
        
        # Get or create root using canonical storage
        root = self.get_or_create_node(board)
        
        # Always run full MCTS - no shortcuts
        if board.isEnd() == Result.Incomplete:
            if not root.is_expanded():
                self.expand(root)
            
            for sim in range(num_simulations):
                self._simulate_once(root, board.get_depth())
                
        return root
    
    def expand(self, node):
        """Expand ALL valid moves"""
        policy, value = self.get_policy_value(node.original_board)
        valid_moves = node.original_board.get_valid_moves()
        
        if not valid_moves:
            return
        
        # Mask invalid moves
        policy_masked = np.zeros(BOARD_SIZE)
        policy_masked[valid_moves] = policy[valid_moves]
        
        # Normalize
        policy_sum = policy_masked.sum()
        if policy_sum > 0:
            policy_masked /= policy_sum
        else:
            policy_masked[valid_moves] = 1.0 / len(valid_moves)
        
        # Get current player
        current_player = node.original_board.next_player()
        
        # Create children for ALL valid moves
        for move in valid_moves:
            new_board = node.original_board.act(move, current_player)
            
            # Always create logical child relationship
            child = self.get_or_create_node(new_board, parent=node, parent_move=move, 
                                          prior=policy_masked[move])
            node.children[move] = child
    
    def _simulate_once(self, root, start_depth):
        """Single MCTS simulation"""
        node = root
        path = [node]
        current_depth = start_depth
        
        # Selection phase
        while node.is_expanded() and len(node.children) > 0:
            node = self.select_child(node)
            path.append(node)
            current_depth += 1
        
        # Expansion & Evaluation
        result = node.original_board.isEnd()
        
        if result == Result.Incomplete and not node.is_expanded():
            self.expand(node)
            # Select random child for simulation
            if node.children:
                moves = list(node.children.keys())
                move = np.random.choice(moves)
                node = node.children[move]
                path.append(node)
                current_depth += 1
                result = node.original_board.isEnd()
        
        # Get value
        if result != Result.Incomplete:
            # Use board's next_player to determine perspective
            current_player = Cell.X if current_depth % 2 == 0 else Cell.O
            value = self.get_terminal_value(result, current_player)
        else:
            _, value = self.get_policy_value(node.original_board)
        
        # Backpropagate
        self.backpropagate(path, value)

    def select_child(self, node):
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            score = child.ucb_score(node.visits)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def get_terminal_value(self, result, current_player):
        """Get value for terminal states from current player's perspective"""
        if result == Result.Draw:
            return 0
        elif result == Result.X_Wins:
            return 1 if current_player == Cell.X else -1
        else:  # O wins
            return 1 if current_player == Cell.O else -1
    
    def backpropagate(self, search_path, value):
        """Backpropagate value up the tree"""
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
    
    def get_action_probs(self, root, temperature=1.0):
        """Get action probabilities"""
        visits = np.zeros(BOARD_SIZE)
        
        if len(root.children) == 0:
            print("WARNING: Root has no children!")
            return visits
        
        for move, child in root.children.items():
            visits[move] = child.visits
        
        if temperature == 0:
            # Greedy
            probs = np.zeros(BOARD_SIZE)
            best_move = np.argmax(visits)
            probs[best_move] = 1.0
        else:
            # Temperature scaling
            if visits.sum() == 0:
                # Fallback to uniform
                valid_moves = list(root.children.keys())
                probs = np.zeros(BOARD_SIZE)
                for move in valid_moves:
                    probs[move] = 1.0 / len(valid_moves)
            else:
                visits_temp = visits ** (1.0 / temperature)
                probs = visits_temp / visits_temp.sum()
        
        return probs

class NeuralMCTSPlayer:
    def __init__(self, model=None, model_path=None, num_simulations=800, temperature=1.0, device='cpu'):
        self.symbol = None
        self.ai_symbol = None
        self.state = None
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.device = device
        
        # Use provided model OR create new one
        if model is not None:
            self.model = model
        else:
            # Initialize NEW model
            self.model = SimpleCNN(row=BOARD_ROWS, col=BOARD_COLS, in_chan=3)
            
            # Load weights if path provided
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded model from {model_path}")
        
        self.model.to(device)
        self.mcts = NeuralMCTS(self.model, device)

    def reset(self):
        self.state = None
    
    def set_state(self, state):
        self.state = state
    
    def setSymbol(self, symbol):
        self.symbol = symbol
        self.ai_symbol = symbol 
    
    def act(self, temperature=None, training=False):
        """Select action using MCTS - returns (move, symbol) for compatibility"""
        if temperature is None:
            temperature = self.temperature
            
        # Make sure MCTS uses AI's perspective
        root = self.mcts.search(self.state, self.num_simulations, ai_symbol=self.ai_symbol)
        action_probs = self.mcts.get_action_probs(root, temperature)
        
        # Sample action from distribution
        valid_moves = self.state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
            
        valid_probs = action_probs[valid_moves]
        if valid_probs.sum() == 0:
            # Fallback to uniform
            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)
        else:
            valid_probs /= valid_probs.sum()
        
        if training or temperature > 0:
            move = np.random.choice(valid_moves, p=valid_probs)
        else:
            # Greedy for evaluation
            move = valid_moves[np.argmax(valid_probs)]
        
        return move, self.symbol

# Training data collection
class TrainingExample:
    def __init__(self, state, policy, value):
        self.state = state  # Board state tensor
        self.policy = policy  # MCTS visit distribution
        self.value = value  # Game outcome from this player's perspective

def collect_self_play_data(model, num_games=100, num_simulations=400, device='cpu'):
    """Collect training data through self-play"""
    training_examples = []
    
    for game_idx in range(num_games):
        game_examples = []
        board = Board()
        
        # Single player that switches perspective
        player = NeuralMCTSPlayer(model=model, num_simulations=num_simulations, 
                                 temperature=1.0, device=device)
        
        while True:
            current_symbol = board.next_player()
            player.setSymbol(current_symbol)
            player.set_state(board)
            
            # Get state from current player's perspective
            state_tensor = player.mcts.board_to_tensor(board)
            
            # Run MCTS
            root = player.mcts.search(board, num_simulations, ai_symbol=current_symbol)
            action_probs = player.mcts.get_action_probs(root, temperature=1.0)
            
            # Store example (value will be filled later)
            game_examples.append((state_tensor.cpu(), action_probs, current_symbol))
            
            # Make move
            valid_moves = board.get_valid_moves()
            valid_probs = action_probs[valid_moves]
            if valid_probs.sum() > 0:
                valid_probs /= valid_probs.sum()
                move = np.random.choice(valid_moves, p=valid_probs)
            else:
                move = np.random.choice(valid_moves)
            
            board = board.act(move, current_symbol)
            result = board.isEnd()
            
            if result != Result.Incomplete:
                # Assign values from each move's current player perspective
                for i, (state, policy, move_player) in enumerate(game_examples):
                    if result == Result.Draw:
                        value = 0
                    elif result == Result.X_Wins:
                        value = 1 if move_player == Cell.X else -1
                    else:  # O wins
                        value = 1 if move_player == Cell.O else -1
                    
                    training_examples.append(TrainingExample(state, policy, value))
                break
        
        if (game_idx + 1) % 10 == 0:
            print(f"Collected {game_idx + 1}/{num_games} games")
    
    return training_examples

def train_network(model, training_examples, epochs=10, batch_size=32, lr=0.001, device='cuda'):
    """Train the neural network"""
    model.to(device)
    model.train()
    
    print(f"  Training on {len(training_examples)} examples")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses_history = []
    
    for epoch in range(epochs):
        np.random.shuffle(training_examples)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        num_samples = len(training_examples)
        num_full_batches = max(1, num_samples // batch_size)
        
        for i in range(0, num_full_batches * batch_size, batch_size):
            batch = training_examples[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            states = torch.cat([ex.state for ex in batch]).to(device)
            target_policies = torch.FloatTensor(np.array([ex.policy for ex in batch])).to(device)
            target_values = torch.FloatTensor(np.array([[ex.value] for ex in batch])).to(device)
            
            pred_policies, pred_values = model(states)
            
            # Policy loss: cross-entropy
            policy_loss = -(target_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
            
            # Value loss: MSE
            value_loss = nn.MSELoss()(pred_values, target_values)
            
            # Combined loss
            loss = policy_loss + 3.0 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            losses_history.append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                      f"Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")
    
    return model, losses_history

def evaluate_model(model, opponent_model=None, num_games=50, device='cpu'):
    """Evaluate model performance"""
    if opponent_model is None:
        opponent_model = model  # Self-play evaluation
    
    player1 = NeuralMCTSPlayer(model=model, num_simulations=400, temperature=0.0, device=device)
    player2 = NeuralMCTSPlayer(model=opponent_model, num_simulations=400, temperature=0.0, device=device)
    
    wins = 0
    draws = 0
    losses = 0
    
    for game in range(num_games):
        board = Board()
        
        # Alternate who goes first
        if game % 2 == 0:
            p1, p2 = player1, player2
            p1_is_model = True
        else:
            p1, p2 = player2, player1
            p1_is_model = False
        
        p1.setSymbol(Cell.X)
        p2.setSymbol(Cell.O)
        current_player = p1
        
        while True:
            current_player.set_state(board)
            move, symbol = current_player.act()
            
            board = board.act(move, symbol)
            result = board.isEnd()
            
            if result != Result.Incomplete:
                if result == Result.Draw:
                    draws += 1
                elif (result == Result.X_Wins and p1_is_model) or \
                     (result == Result.O_Wins and not p1_is_model):
                    wins += 1
                else:
                    losses += 1
                break
            
            current_player = p2 if current_player == p1 else p1
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    
    print(f"Evaluation: {wins}W/{draws}D/{losses}L (Win rate: {win_rate:.1%}, Draw rate: {draw_rate:.1%})")
    return win_rate, draw_rate

def train_alphazero_style(iterations=10, games_per_iter=100, epochs_per_iter=10, device='cpu'):
    """Train using AlphaZero-style self-play"""
    model = SimpleCNN(row=BOARD_ROWS, col=BOARD_COLS, in_chan=3).to(device)
    
    print("Starting AlphaZero-style training...")
    print("=" * 50)
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        print("-" * 30)
        
        # Collect self-play data
        print("Collecting self-play data...")
        examples = collect_self_play_data(
            model, 
            num_games=games_per_iter,
            num_simulations=400,
            device=device
        )
        
        # Train network
        print(f"Training network on {len(examples)} examples...")
        model, losses = train_network(
            model, examples,
            epochs=epochs_per_iter,
            batch_size=32,
            lr=0.001,
            device=device
        )
        
        # Evaluate every few iterations
        if (iteration + 1) % 3 == 0:
            print("Evaluating model...")
            evaluate_model(model, device=device)
        
        # Save model
        model_path = f'../policy/alphazero_iter_{iteration}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    
    return model