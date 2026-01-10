import numpy as np
import pickle
import os
from math import sqrt, log
from enum import IntEnum
import sys

# Add parent directory to path so this file can be run directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from strat.encode import Board, Cell, Result
from strat import config

C = sqrt(2)  # UCB exploration constant
PATH = os.path.dirname(os.path.abspath(__file__)) 

class MCTS:
    MAX_NODES = 1000000

    class Node:
        def __init__(self, board, player_to_move):
            self.board = board
            self.player_to_move = player_to_move
            self.value_sum = 0.0
            self.visits = 0
            self.children = {}
            self.parent = None
            self.parent_move = None

        def isFullyExpand(self):
            return len(self.children) == len(self.board.get_valid_moves())

        def __getstate__(self):
            s = self.__dict__.copy()
            s["parent"] = None
            return s

        def __setstate__(self, s):
            self.__dict__.update(s)

    def __init__(self):
        self.nodes = {}

    def get_createNode(self, board: Board, current_symbol) -> 'MCTS.Node':
        hash_val = board.hash()
        if hash_val not in self.nodes:
            if len(self.nodes) >= self.MAX_NODES:
                print(f"Warning: Node limit reached ({self.MAX_NODES})")
                return MCTS.Node(board, current_symbol)
            self.nodes[hash_val] = MCTS.Node(board, current_symbol)
        return self.nodes[hash_val]

    def select(self, node):
        # Choose child that maximizes the CURRENT node's player's value.
        best_score = -1e9
        best_move, best_child = None, None
        for move, child in node.children.items():
            if child.visits == 0:
                exploitation = 1.0
                exploration = 1e3  # encourage first visit
            else:
                child_mean = child.value_sum / child.visits  # win rate for CHILD player_to_move
                exploitation = 1.0 - child_mean            # convert to PARENT perspective
                exploration = C * sqrt(log(node.visits + 1) / child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

    def _immediate_wins(self, board: Board, player: Cell):
        """Return the set of moves that give 'player' an immediate win on 'board'."""
        wins = set()
        target = Result.X_Wins if player == Cell.X else Result.O_Wins
        for mv in board.get_valid_moves():
            if board.act(mv, player).isEnd() == target:
                wins.add(mv)
        return wins

    def _threat_moves(self, board: Board, player: Cell):
        """Return moves that create threats for the player."""
        threats = set()
        board_2d = board.visualize()
        
        # For 3x3 tic-tac-toe, look for 2-in-a-row opportunities
        sequences = board.get_rows_cols_and_diagonals()
        valid_moves = board.get_valid_moves()
        
        for move in valid_moves:
            test_board = board.act(move, player)
            test_sequences = test_board.get_rows_cols_and_diagonals()
            
            # Check if this move creates a 2-in-a-row
            target_sum = 2 if player == Cell.X else -2
            if target_sum in test_sequences:
                threats.add(move)
        
        return threats

    def expand(self, node: 'MCTS.Node', board: Board, current_symbol):
        unexplored = [m for m in board.get_valid_moves() if m not in node.children]
        if not unexplored:
            return None
            
        priorities = []
        opponent = Cell.O if current_symbol == Cell.X else Cell.X

        # Get immediate tactical information
        my_wins_now = self._immediate_wins(board, current_symbol)
        opp_wins_now = self._immediate_wins(board, opponent)
        my_threats = self._threat_moves(board, current_symbol)

        for m in unexplored:
            new_board = board.act(m, current_symbol)
            priority = 0

            # Check if move ends game
            end = new_board.isEnd()
            if end != Result.Incomplete:
                if (end == Result.X_Wins and current_symbol == Cell.X) or \
                   (end == Result.O_Wins and current_symbol == Cell.O):
                    priority = 20  # Win immediately
                else:
                    priority = -10  # Lose
            else:
                # Immediate win (highest priority)
                if m in my_wins_now:
                    priority = max(priority, 20)
                
                # Block opponent win (very high priority)
                elif m in opp_wins_now:
                    priority = max(priority, 18)
                
                # Create threat (medium priority)
                elif m in my_threats:
                    priority = max(priority, 8)
                
                # Prefer center and corners for 3x3
                if m == 4:  # Center
                    priority = max(priority, 6)
                elif m in [0, 2, 6, 8]:  # Corners
                    priority = max(priority, 4)

                # Penalize moves that still leave opponent immediate win
                if self._immediate_wins(new_board, opponent):
                    priority = min(priority, -5)

            priorities.append((priority, -np.random.random(), m))

        priorities.sort(reverse=True)
        move = priorities[0][2]
        next_symbol = opponent
        new_board = board.act(move, current_symbol)
        child_node = self.get_createNode(new_board, next_symbol)
        child_node.parent = node
        child_node.parent_move = move
        node.children[move] = child_node
        return move

    def simulate(self, board, starting_symbol):
        """Random simulation with tactical awareness"""
        curr = Board(board.cells.copy())
        symbol = starting_symbol
        
        for _ in range(20):  # Prevent infinite loops
            result = curr.isEnd()
            if result != Result.Incomplete:
                return result

            valid_moves = curr.get_valid_moves()
            if not valid_moves:
                return Result.Draw
                
            priorities = []
            opponent = Cell.O if symbol == Cell.X else Cell.X

            # Get tactical information
            my_wins = self._immediate_wins(curr, symbol)
            opp_wins = self._immediate_wins(curr, opponent)

            for mv in valid_moves:
                test_board = curr.act(mv, symbol)
                test_result = test_board.isEnd()
                priority = 0

                if test_result != Result.Incomplete:
                    if (test_result == Result.X_Wins and symbol == Cell.X) or \
                       (test_result == Result.O_Wins and symbol == Cell.O):
                        priority = 20
                    else:
                        priority = -10
                else:
                    # Win immediately
                    if mv in my_wins:
                        priority = 20
                    # Block opponent win
                    elif mv in opp_wins:
                        priority = 18
                    # Prefer center and corners
                    elif mv == 4:  # Center
                        priority = 6
                    elif mv in [0, 2, 6, 8]:  # Corners
                        priority = 4
                    else:
                        priority = 1

                priorities.append((priority, -np.random.random(), mv))

            priorities.sort(reverse=True)
            mv = priorities[0][2]
            curr = curr.act(mv, symbol)
            symbol = opponent
        
        return Result.Draw  # Fallback if simulation runs too long

    def backpropagate(self, node, result):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            if result == Result.Draw:
                value = 0.5
            elif (result == Result.X_Wins and current_node.player_to_move == Cell.X) or \
                 (result == Result.O_Wins and current_node.player_to_move == Cell.O):
                value = 1.0
            else:
                value = 0.0
            current_node.value_sum += value
            current_node = current_node.parent

    def getBestMove(self, node, temperature=0.0, training=False):
        board = node.board
        current_symbol = node.player_to_move
        opponent = Cell.O if current_symbol == Cell.X else Cell.X

        # Always take immediate wins
        my_wins = self._immediate_wins(board, current_symbol)
        if my_wins:
            return min(my_wins)

        # Always block opponent wins
        opp_wins = self._immediate_wins(board, opponent)
        if opp_wins:
            return min(opp_wins)

        if not node.children:
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            # Prefer center, then corners
            if 4 in valid_moves:
                return 4
            corners = [m for m in valid_moves if m in [0, 2, 6, 8]]
            if corners:
                return np.random.choice(corners)
            return np.random.choice(valid_moves)

        if training:
            # Most visited during training
            return max(node.children.items(), key=lambda kv: kv[1].visits)[0]

        moves = list(node.children.keys())
        child_means = np.array([
            node.children[m].value_sum / max(1, node.children[m].visits) for m in moves
        ])
        parent_values = 1.0 - child_means  # Convert to current player's perspective

        if temperature == 0:
            return moves[np.argmax(parent_values)]
        else:
            noisy_values = parent_values + np.random.normal(0, temperature, len(parent_values))
            return moves[np.argmax(noisy_values)]

class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.swap_players = False 
        self.reset_symbols()

    def reset_symbols(self):
        if self.swap_players:
            self.p1.setSymbol(Cell.O)
            self.p2.setSymbol(Cell.X)
        else:
            self.p1.setSymbol(Cell.X)
            self.p2.setSymbol(Cell.O)    

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        if self.swap_players:
            while True:
                yield self.p2
                yield self.p1
        else:
            while True:
                yield self.p1
                yield self.p2
        
    def play(self, print_state=False):
        self.reset_symbols()
        alternator = self.alternate()
        self.reset()
        current_state = Board()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print()
            
        for move_count in range(9):  # Max 9 moves in 3x3
            player = next(alternator)
            move = player.act(training=True)
            current_state = current_state.act(move, player.symbol)
            isend = current_state.isEnd()
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print()
            if isend != Result.Incomplete:
                return isend, player
        
        return Result.Draw, None
        
    def play_with_temp(self, temperature=0.0, print_state=False):
        self.reset_symbols()
        alternator = self.alternate()
        self.reset()
        current_state = Board()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print()
            
        for move_count in range(9):  # Max 9 moves in 3x3
            player = next(alternator)
            move = player.act(temperature=temperature, training=True)
            current_state = current_state.act(move, player.symbol)
            isend = current_state.isEnd()
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print()
            if isend != Result.Incomplete:
                return isend, player
        
        return Result.Draw, None

class MCTSPlayer:
    _shared_mcts = None
    
    def __init__(self, num_sim=5000, use_shared_tree=True, trained_mode=False):
        self.symbol = None
        self.state = None
        self.num_simulations = num_sim
        self.trained_mode = trained_mode
        if use_shared_tree:
            if MCTSPlayer._shared_mcts is None:
                MCTSPlayer._shared_mcts = MCTS()
            self.mcts = MCTSPlayer._shared_mcts
        else:
            self.mcts = MCTS()

    def reset(self):
        self.state = None
    
    def set_state(self, state):
        self.state = state
    
    def setSymbol(self, symbol):
        self.symbol = symbol 
    
    def act(self, temperature=0.0, training=False):
        board = self.state
        root = self.mcts.get_createNode(self.state, self.symbol)
        
        # Use cached results if available and trained
        if self.trained_mode and root.visits > 1000:
            best_move = self.mcts.getBestMove(root, 0.0, training=True)
            return best_move
        
        # Run MCTS simulations
        num_sims = self.num_simulations if not self.trained_mode else 2000
        for _ in range(num_sims):
            node = root
            sim_board = board
            current_symbol = self.symbol
            
            # Selection phase
            while node.isFullyExpand() and len(node.children) > 0:
                move, node = self.mcts.select(node)
                sim_board = sim_board.act(move, current_symbol)
                current_symbol = Cell.O if current_symbol == Cell.X else Cell.X
            
            # Check if simulation ends
            result = sim_board.isEnd()
            if result == Result.Incomplete and not node.isFullyExpand():
                # Expansion
                move = self.mcts.expand(node, sim_board, current_symbol)
                if move is None:
                    continue
                sim_board = sim_board.act(move, current_symbol)
                node = node.children[move]
                current_symbol = Cell.O if current_symbol == Cell.X else Cell.X
            
            # Simulation phase
            result = self.mcts.simulate(sim_board, current_symbol)
            
            # Backpropagation
            self.mcts.backpropagate(node, result)
        
        best_move = self.mcts.getBestMove(root, temperature, training)
        return best_move

    def save_policy(self):
        with open(f'{PATH}/mcts_tree.bin', 'wb') as f:
            pickle.dump(self.mcts.nodes, f)
        print(f"Saved {len(self.mcts.nodes)} nodes")
    
    def load_policy(self):
        try:
            with open(f'{PATH}/mcts_tree.bin', 'rb') as f:
                self.mcts.nodes = pickle.load(f)
            # Restore parent-child relationships
            for node in self.mcts.nodes.values():
                for move, child in node.children.items():
                    child.parent = node
                    child.parent_move = move
            print(f"Loaded MCTS policy with {len(self.mcts.nodes)} nodes")
        except FileNotFoundError:
            print("No saved MCTS policy found, starting fresh.")
    
    @classmethod
    def clear_shared_tree(cls):
        cls._shared_mcts = None

class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def setSymbol(self, symbol):
        self.symbol = symbol

    def act(self, *args, **kwargs):
        self.state.print()
        print("Positions:")
        print("0 1 2")
        print("3 4 5")
        print("6 7 8")
        
        while True:
            try:
                move_input = input("Enter move (0-8): ")
                move = int(move_input)
                if 0 <= move <= 8:
                    if self.state.cells[move] == Cell.Empty:
                        return move
                    else:
                        print("Position already taken!")
                else:
                    print("Invalid position! Use 0-8")
            except (ValueError, IndexError):
                print("Invalid input! Enter a number 0-8")

def train(epochs, print_every_n=10):
    """Train MCTS on 3x3 tic-tac-toe"""
    MCTSPlayer.clear_shared_tree()
    player1 = MCTSPlayer(num_sim=3000, use_shared_tree=True, trained_mode=False)
    player2 = MCTSPlayer(num_sim=3000, use_shared_tree=True, trained_mode=False)
    judger = Judger(player1, player2)
    
    x_wins = 0
    o_wins = 0
    draws = 0
    
    print("Training MCTS on 3x3 Tic-Tac-Toe...")
    
    for i in range(1, epochs + 1):
        if i % 2 == 0:
            judger.swap_players = not judger.swap_players
        
        temp = 0.1 if i < epochs // 4 else 0.0  # Exploration early, exploitation later
        result, winner = judger.play_with_temp(temperature=temp, print_state=False)
        
        if result == Result.Draw:
            draws += 1
        elif result == Result.X_Wins:
            x_wins += 1
        elif result == Result.O_Wins:
            o_wins += 1
        
        if i % print_every_n == 0:
            print(f"Epoch {i}: X wins: {x_wins/i:.2%}, O wins: {o_wins/i:.2%}, Draws: {draws/i:.2%}")
            print(f"   Total nodes: {len(player1.mcts.nodes)}")
        
        judger.reset()
    
    player1.save_policy()
    print(f"Training complete! Total nodes: {len(player1.mcts.nodes)}")

def play():
    """Play against trained MCTS"""
    human = HumanPlayer()
    mcts = MCTSPlayer(num_sim=5000, trained_mode=True)
    judger = Judger(mcts, human)
    
    try:
        mcts.load_policy()
        print("Loaded trained MCTS policy")
    except:
        print("No trained policy found, using fresh MCTS")
    
    print("\nYou are O, AI is X")
    result, winner = judger.play(print_state=True)
    
    if result == Result.X_Wins:
        print("AI (X) wins!")
    elif result == Result.O_Wins:
        print("You (O) win!") 
    else:
        print("Draw!")

def test_tactical():
    """Test MCTS tactical awareness"""
    print("Testing tactical awareness...")
    
    # Test blocking
    board = Board(np.array([Cell.X, Cell.X, Cell.Empty,
                           Cell.O, Cell.Empty, Cell.Empty,
                           Cell.Empty, Cell.Empty, Cell.Empty]))
    
    player = MCTSPlayer(num_sim=1000, use_shared_tree=False)
    player.setSymbol(Cell.O)
    player.set_state(board)
    
    move = player.act(temperature=0, training=False)
    expected_move = 2  # Should block X's win
    
    if move == expected_move:
        print("✅ Blocking test passed!")
    else:
        print(f"❌ Blocking test failed! Expected {expected_move}, got {move}")
    
    # Test winning
    board2 = Board(np.array([Cell.O, Cell.O, Cell.Empty,
                            Cell.X, Cell.Empty, Cell.Empty,
                            Cell.X, Cell.Empty, Cell.Empty]))
    
    player.set_state(board2)
    move2 = player.act(temperature=0, training=False)
    expected_move2 = 2  # Should win
    
    if move2 == expected_move2:
        print("✅ Winning test passed!")
    else:
        print(f"❌ Winning test failed! Expected {expected_move2}, got {move2}")

if __name__ == '__main__':
    print("3x3 Tic-Tac-Toe MCTS")
    print("=" * 30)
    
    # Test tactical awareness first
    test_tactical()
    
    # Train the agent
    train(1000, print_every_n=50)
    play()