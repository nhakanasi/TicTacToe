import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
strat_dir = os.path.join(current_dir, 'strat')

sys.path.append(parent_dir)
sys.path.append(current_dir)
sys.path.append(strat_dir)
from strat.mcts_neural_alpha_go import *
from model.simplecnn import SimpleCNN
from strat.alpha_beta import minimax
import numpy as np

# Initialize model and device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN(row=BOARD_ROWS, col=BOARD_COLS, in_chan=3).to(device)

class AlphaBetaPlayer:
    """Alpha-beta minimax player using your implementation"""
    
    def __init__(self, max_depth=9):
        self.symbol = None
        self.max_depth = max_depth
        self.board = None
    
    def reset(self):
        pass
    
    def set_state(self, board):
        self.board = board
    
    def setSymbol(self, symbol):
        self.symbol = symbol
    
    def act(self, board=None):
        """Find best move using alpha-beta minimax"""
        if board is None:
            board = self.board
            
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return None
        
        best_move = None
        best_value = -999
        
        for move in valid_moves:
            # Try this move
            child_board = board.act(move, self.symbol)
            
            # Get opponent symbol
            opponent = Cell.O if self.symbol == Cell.X else Cell.X
            
            # Evaluate using your minimax function
            value = minimax(child_board, -999, 999, opponent, self.symbol)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move

class PerfectTacticalMCTS(NeuralMCTS):
    
    def __init__(self, model, device='cpu'):
        super().__init__(model, device)
        self.forced_moves = 0
        self.total_moves = 0
    
    def get_winning_move(self, board):
        """Find immediate winning move"""
        current_player = board.next_player()
        valid_moves = board.get_valid_moves()
        
        for move in valid_moves:
            test_board = board.act(move, current_player)
            result = test_board.isEnd()
            expected_result = Result.X_Wins if current_player == Cell.X else Result.O_Wins
            if result == expected_result:
                return move
        return None
    
    def get_blocking_move(self, board):
        """Find move to block opponent win"""
        current_player = board.next_player()
        opponent = Cell.X if current_player == Cell.O else Cell.O
        valid_moves = board.get_valid_moves()
        
        for move in valid_moves:
            test_board = board.act(move, opponent)
            result = test_board.isEnd()
            expected_result = Result.X_Wins if opponent == Cell.X else Result.O_Wins
            if result == expected_result:
                return move
        return None
    
    def search(self, board: Board, num_simulations=800, ai_symbol=None):
        """Search with tactical forcing"""
        if ai_symbol is None:
            ai_symbol = board.next_player()
        
        self.ai_symbol = ai_symbol
        self.total_moves += 1
        
        # Check for tactical moves first
        winning_move = self.get_winning_move(board)
        if winning_move is not None:
            self.forced_moves += 1
            return self.create_forced_root(board, winning_move, 'win')
        
        blocking_move = self.get_blocking_move(board)
        if blocking_move is not None:
            self.forced_moves += 1
            return self.create_forced_root(board, blocking_move, 'block')
        
        # Normal MCTS search
        return super().search(board, num_simulations, ai_symbol)
    
    def create_forced_root(self, board, forced_move, move_type):
        """Create root that forces tactical move"""
        root = self.get_or_create_node(board)
        valid_moves = board.get_valid_moves()
        current_player = board.next_player()
        
        for move in valid_moves:
            new_board = board.act(move, current_player)
            child = self.get_or_create_node(new_board, parent=root, parent_move=move)
            
            if move == forced_move:
                child.visits = 1000
                child.value_sum = 900 if move_type == 'win' else 700
                child.prior = 0.9
            else:
                child.visits = 1
                child.value_sum = -10
                child.prior = 0.1 / (len(valid_moves) - 1)
            
            root.children[move] = child
        
        root.visits = sum(child.visits for child in root.children.values())
        return root

def collect_training_vs_alphabeta(model, num_games=100, device='cpu'):
    """Collect training data by playing neural network vs alpha-beta"""
    
    training_examples = []
    neural_wins = 0
    draws = 0
    losses = 0
    
    # Create neural player with tactical forcing
    neural_player = NeuralMCTSPlayer(
        model=model,
        num_simulations=50,
        temperature=0.3,
        device=device
    )
    neural_player.mcts = PerfectTacticalMCTS(model, device)
    
    # Create alpha-beta player
    alphabeta_player = AlphaBetaPlayer(max_depth=8)
    
    for game_idx in range(num_games):
        game_examples = []
        board = Board()
        
        # Alternate who goes first
        if game_idx % 2 == 0:
            neural_player.setSymbol(Cell.X)
            alphabeta_player.setSymbol(Cell.O)
            current_player = neural_player
            neural_is_x = True
        else:
            neural_player.setSymbol(Cell.O) 
            alphabeta_player.setSymbol(Cell.X)
            current_player = alphabeta_player
            neural_is_x = False
        
        move_count = 0
        max_moves = 20
        
        # Play game
        while move_count < max_moves:
            # Set states
            neural_player.set_state(board)
            alphabeta_player.set_state(board)
            
            # Collect training data from neural player
            if current_player == neural_player:
                state_tensor = neural_player.mcts.board_to_tensor(board)
                root = neural_player.mcts.search(board, neural_player.num_simulations)
                action_probs = neural_player.mcts.get_action_probs(root, neural_player.temperature)
                player_symbol = board.next_player()   # ALWAYS matches tensor perspective
                game_examples.append((state_tensor.cpu(), action_probs, player_symbol))

            
            # Make move
            if current_player == alphabeta_player:
                move = current_player.act(board)
                symbol = current_player.symbol
            else:
                move, symbol = current_player.act()
            
            if move is None:
                break
                
            board = board.act(move, symbol)
            result = board.isEnd()
            move_count += 1
            
            if result != Result.Incomplete:
                # Count results from neural player's perspective
                if result == Result.Draw:
                    draws += 1
                elif (result == Result.X_Wins and neural_is_x) or \
                     (result == Result.O_Wins and not neural_is_x):
                    neural_wins += 1
                else:
                    losses += 1
                
                # Create training examples
                for state, policy, player_symbol in game_examples:
                    if result == Result.Draw:
                        value = 0
                    elif result == Result.X_Wins:
                        value = 1 if player_symbol == Cell.X else -1
                    else:  # O wins
                        value = 1 if player_symbol == Cell.O else -1
                    
                    training_examples.append(TrainingExample(state, policy, value))
                break
            
            # Switch players
            current_player = alphabeta_player if current_player == neural_player else neural_player
    
    performance = (neural_wins + draws) / num_games
    return training_examples, performance

def evaluate_vs_alphabeta(model, games=50, device='cpu'):
    """Evaluate neural network vs alpha-beta"""
    
    neural_player = NeuralMCTSPlayer(
        model=model,
        num_simulations=100,
        temperature=0.0,
        device=device
    )
    neural_player.mcts = PerfectTacticalMCTS(model, device)
    
    alphabeta_player = AlphaBetaPlayer(max_depth=9)
    
    neural_wins = 0
    draws = 0
    losses = 0
    
    for game in range(games):
        board = Board()
        
        # Alternate starting player
        if game % 2 == 0:
            p1, p2 = neural_player, alphabeta_player
            neural_is_x = True
        else:
            p1, p2 = alphabeta_player, neural_player
            neural_is_x = False
            
        p1.setSymbol(Cell.X)
        p2.setSymbol(Cell.O)
        current_player = p1
        
        move_count = 0
        while move_count < 20:
            if hasattr(current_player, 'set_state'):
                current_player.set_state(board)
                
            if isinstance(current_player, AlphaBetaPlayer):
                move = current_player.act(board)
            else:
                move, _ = current_player.act()
            
            if move is None:
                break
                
            board = board.act(move, current_player.symbol)
            result = board.isEnd()
            move_count += 1
            
            if result != Result.Incomplete:
                if result == Result.Draw:
                    draws += 1
                elif (result == Result.X_Wins and neural_is_x) or \
                     (result == Result.O_Wins and not neural_is_x):
                    neural_wins += 1
                else:
                    losses += 1
                break
                
            current_player = p2 if current_player == p1 else p1
    
    win_rate = neural_wins / games
    draw_rate = draws / games
    loss_rate = losses / games
    
    return win_rate, draw_rate, loss_rate

def evaluate_tactical_strength(model, device='cpu'):
    """Test if neural network learned basic tactics"""
    
    test_cases = [
        {
            'board': np.array([Cell.X, Cell.Empty, Cell.Empty,
                              Cell.Empty, Cell.X, Cell.Empty,
                              Cell.Empty, Cell.Empty, Cell.Empty]),
            'player': Cell.O,
            'correct_moves': [8],
        },
        {
            'board': np.array([Cell.O, Cell.O, Cell.Empty,
                              Cell.X, Cell.Empty, Cell.Empty,
                              Cell.Empty, Cell.X, Cell.Empty]),
            'player': Cell.O,
            'correct_moves': [2],
        },
        {
            'board': np.array([Cell.Empty, Cell.Empty, Cell.Empty,
                              Cell.X, Cell.X, Cell.Empty,
                              Cell.O, Cell.Empty, Cell.Empty]),
            'player': Cell.O,
            'correct_moves': [5],
        }
    ]
    
    # Test WITH tactical forcing
    player_with_forcing = NeuralMCTSPlayer(model=model, num_simulations=200, temperature=0.0, device=device)
    player_with_forcing.mcts = PerfectTacticalMCTS(model, device)
    
    passed_with_forcing = 0
    for test in test_cases:
        board = Board(test['board'])
        player_with_forcing.setSymbol(test['player'])
        player_with_forcing.set_state(board)
        move, _ = player_with_forcing.act()
        if move in test['correct_moves']:
            passed_with_forcing += 1
    
    forcing_success = passed_with_forcing / len(test_cases)
    
    # Test WITHOUT forcing (pure neural network)
    player_pure = NeuralMCTSPlayer(model=model, num_simulations=400, temperature=0.0, device=device)
    
    passed_pure = 0
    for test in test_cases:
        board = Board(test['board'])
        player_pure.setSymbol(test['player'])
        player_pure.set_state(board)
        move, _ = player_pure.act()
        if move in test['correct_moves']:
            passed_pure += 1
    
    learned_rate = passed_pure / len(test_cases)
    
    return forcing_success, learned_rate

def train_with_alphabeta(epochs=12):
    """Train neural network against alpha-beta minimax"""
    global model
    
    print("Training Neural Network vs Alpha-Beta Minimax")
    print("=" * 50)
    
    for iteration in range(epochs):
        print(f"Iteration {iteration + 1}/{epochs}")
        
        # Collect training data
        num_games = 60 + (iteration * 8)
        examples, performance = collect_training_vs_alphabeta(
            model, num_games=num_games, device=device
        )
        
        print(f"Collected {len(examples)} examples, Performance: {performance:.3f}")
        
        # Training parameters
        if iteration < 4:
            lr, epochs_per_iter, batch_size = 0.025, 18, 32
        elif iteration < 8:
            lr, epochs_per_iter, batch_size = 0.02, 14, 48
        else:
            lr, epochs_per_iter, batch_size = 0.015, 10, 64
        
        # Train network
        model, losses = train_network(
            model, examples,
            epochs=epochs_per_iter,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        
        # Evaluate every 2 iterations
        if (iteration + 1) % 2 == 0:
            forcing_success, learned_rate = evaluate_tactical_strength(model, device)
            win_rate, draw_rate, loss_rate = evaluate_vs_alphabeta(model, 30, device)
            
            print(f"Progress: Learned {learned_rate:.1%}, Draw {draw_rate:.1%}")
            
            if learned_rate >= 1.0:
                print("Perfect tactical learning achieved!")
                break
        
        # Save model
        model_path = f'../policy/model_alphabeta_iter_{iteration}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

def train_self_play(epochs=8):
    """Train using pure self-play"""
    global model
    
    print("Training with Self-Play")
    print("=" * 30)
    
    for iteration in range(epochs):
        print(f"Self-Play Iteration {iteration + 1}/{epochs}")
        
        # Collect self-play data
        num_games = 50 + (iteration * 10)
        examples = collect_self_play_data(
            model, 
            num_games=num_games,
            num_simulations=300,
            device=device
        )
        
        print(f"Collected {len(examples)} self-play examples")
        
        # Training parameters
        lr = 0.01 if iteration < 4 else 0.008
        epochs_per_iter = 12 if iteration < 4 else 8
        
        # Train network
        model, losses = train_network(
            model, examples,
            epochs=epochs_per_iter,
            batch_size=48,
            lr=lr,
            device=device
        )
        
        # Evaluate every 2 iterations
        if (iteration + 1) % 2 == 0:
            evaluate_model(model, device=device)
        
        # Save model
        model_path = f'../policy/model_selfplay_iter_{iteration}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

def play_against_ai():
    """Play against the trained neural network"""
    
    # Load best model
    model_files = [
        '../policy/model_alphabeta_iter_9.pth'
    ]
    
    loaded = False
    for model_path in model_files:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded trained model: {model_path}")
            loaded = True
            break
    
    if not loaded:
        print("No trained model found, using random initialization")
    
    # Create AI with tactical forcing
    ai_player = NeuralMCTSPlayer(
        model=model,
        num_simulations=600,
        temperature=0.0,
        device=device
    )
    ai_player.mcts = PerfectTacticalMCTS(model, device)
    
    # Human vs AI game
    board = Board()
    human_symbol = Cell.O
    ai_symbol = Cell.X
    current_player = Cell.X
    
    ai_player.setSymbol(ai_symbol)
    
    print(f"\nYou are {human_symbol}, AI is {ai_symbol}")
    print("Enter moves as: row col (0-2)")
    
    while True:
        board.print()
        
        if current_player == human_symbol:
            try:
                coord_input = input("Your move (row col): ")
                row, col = map(int, coord_input.split())
                if 0 <= row < 3 and 0 <= col < 3:
                    move = row * 3 + col
                    if board.isValid(move):
                        board = board.act(move, human_symbol)
                        current_player = ai_symbol
                    else:
                        print("Position taken. Try again.")
                else:
                    print("Invalid coordinates. Use 0-2.")
            except (ValueError, IndexError):
                print("Invalid input. Use: row col")
        else:
            ai_player.set_state(board)
            move, _ = ai_player.act()
            board = board.act(move, ai_symbol)
            ai_r, ai_c = divmod(move, 3)
            current_player = human_symbol
        
        result = board.isEnd()
        if result != Result.Incomplete:
            board.print()
            if result == Result.X_Wins:
                print("X won!")
            elif result == Result.O_Wins:
                print("O won!")
            else:
                print("Draw!")
            break

def comprehensive_training():
    """Run comprehensive training pipeline"""
    
    print("Comprehensive Tic-Tac-Toe AI Training Pipeline")
    print("=" * 60)
    
    # Phase 1: Learn tactics vs Alpha-Beta
    print("\nPhase 1: Tactical Learning vs Alpha-Beta")
    train_with_alphabeta(12)
    
    # Phase 2: Self-play refinement
    print("\nPhase 2: Self-Play Refinement")
    train_self_play(8)
    
    # Final evaluation
    print("\nFinal Evaluation")
    print("-" * 30)
    forcing_success, learned_rate = evaluate_tactical_strength(model, device)
    win_rate, draw_rate, loss_rate = evaluate_vs_alphabeta(model, 100, device)
    
    print(f"Tactical Learning: {learned_rate:.1%}")
    print(f"vs Alpha-Beta: Win {win_rate:.1%}, Draw {draw_rate:.1%}, Loss {loss_rate:.1%}")
    
    # Save final model
    final_model_path = '../policy/final_trained_model.pth'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

def train(epochs=12, base_games=60, game_increment=8, lr_early=0.025, lr_mid=0.02, lr_late=0.015, 
          epochs_early=18, epochs_mid=14, epochs_late=10, batch_early=32, batch_mid=48, batch_late=64):
    """
    Train neural network vs alpha-beta with custom parameters
    
    Args:
        epochs: Number of training iterations (default: 12)
        base_games: Starting games per iteration (default: 60)
        game_increment: Additional games each iteration (default: 8)
        lr_early/mid/late: Learning rates for different phases (default: 0.025/0.02/0.015)
        epochs_early/mid/late: Epochs per iteration for different phases (default: 18/14/10)
        batch_early/mid/late: Batch sizes for different phases (default: 32/48/64)
    """
    global model
    
    print(f"Training: {epochs} epochs, {base_games}+{game_increment} games/iter")
    print(f"LR: {lr_early}/{lr_mid}/{lr_late}, Epochs: {epochs_early}/{epochs_mid}/{epochs_late}")
    
    for iteration in range(epochs):
        print(f"Iteration {iteration + 1}/{epochs}")
        
        # Collect training data
        num_games = base_games + (iteration * game_increment)
        examples, performance = collect_training_vs_alphabeta(
            model, num_games=num_games, device=device
        )
        
        print(f"Games: {num_games}, Examples: {len(examples)}, Performance: {performance:.3f}")
        
        # Training parameters based on iteration
        if iteration < 4:
            lr, epochs_per_iter, batch_size = lr_early, epochs_early, batch_early
        elif iteration < 8:
            lr, epochs_per_iter, batch_size = lr_mid, epochs_mid, batch_mid
        else:
            lr, epochs_per_iter, batch_size = lr_late, epochs_late, batch_late
        
        # Train network
        model, losses = train_network(
            model, examples,
            epochs=epochs_per_iter,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        
        # Evaluate every 2 iterations
        if (iteration + 1) % 2 == 0:
            forcing_success, learned_rate = evaluate_tactical_strength(model, device)
            win_rate, draw_rate, loss_rate = evaluate_vs_alphabeta(model, 30, device)
            
            print(f"Progress: Learned {learned_rate:.1%}, Draw {draw_rate:.1%}")
            
            if learned_rate >= 1.0:
                print("Perfect tactical learning achieved!")
                break
        
        # Save model
        model_path = f'../policy/model_alphabeta_iter_{iteration}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    
    # Final evaluation
    print("\nFinal Results:")
    forcing_success, learned_rate = evaluate_tactical_strength(model, device)
    win_rate, draw_rate, loss_rate = evaluate_vs_alphabeta(model, 50, device)
    print(f"Tactical: {learned_rate:.1%}, vs Alpha-Beta: W{win_rate:.1%}/D{draw_rate:.1%}/L{loss_rate:.1%}")

# train(
#     epochs=10,           # Number of iterations
#     base_games=50,       # Starting games per iteration  
#     game_increment=5,    # Additional games each iteration
#     lr_early=0.03,       # Learning rate for first 4 iterations
#     lr_mid=0.02,         # Learning rate for iterations 4-8
#     lr_late=0.01,        # Learning rate for final iterations
#     epochs_early=20,     # Training epochs per iteration (early)
#     epochs_mid=15,       # Training epochs per iteration (mid)
#     epochs_late=10       # Training epochs per iteration (late)
# )
play_against_ai()