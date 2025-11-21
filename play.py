import sys
import os

from strat.mcts import MCTS
from strat.encode import Board, Cell, Result, BOARD_ROWS, BOARD_COLS
import numpy as np

class HumanPlayer:
    def __init__(self):
        self.symbol = None
        self.board = None

    def reset(self):
        pass

    def set_board(self, board):
        self.board = board

    def setSymbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.board.print()
        
        while True:
            try:
                user_input = input("Enter your move (row col): ").strip()
                
                if user_input.lower() in ['quit', 'q', 'exit']:
                    raise KeyboardInterrupt
                
                # Parse input
                parts = user_input.split()
                if len(parts) != 2:
                    print("Invalid input! Please enter row and column separated by space (e.g., '0 1')")
                    continue
                
                row = int(parts[0])
                col = int(parts[1])
                
                # Validate coordinates
                if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
                    print(f"Invalid coordinates! Row and column must be between 0 and {BOARD_ROWS-1}")
                    continue
                
                # Convert to 1D index
                move_1d = row * BOARD_COLS + col
                
                if not self.board.isValid(move_1d):
                    print("Invalid move! Position already taken.")
                    continue
                    
                return move_1d
                
            except ValueError:
                print("Invalid input! Please enter two numbers separated by space (e.g., '0 1')")
                continue
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise

class MCTSPlayer:
    def __init__(self, iterations=2000):
        self.mcts = MCTS()
        self.iterations = iterations
        self.symbol = None

    def reset(self):
        pass

    def set_board(self, board):
        self.board = board

    def setSymbol(self, symbol):
        self.symbol = symbol

    def act(self):
        move = self.mcts.search(self.board, self.iterations)
        # Convert 1D move back to row, col for display
        row, col = divmod(move, BOARD_COLS)
        return move

class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.p1_symbol = Cell.X  # Human plays X (goes first)
        self.p2_symbol = Cell.O  # MCTS plays O
        self.p1.setSymbol(self.p1_symbol)
        self.p2.setSymbol(self.p2_symbol)

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, print_state=True):
        alternator = self.alternate()
        self.reset()
        current_board = Board()
        
        self.p1.set_board(current_board)
        self.p2.set_board(current_board)

        if print_state:
            print("Game starts! You are X, MCTS is O")
            current_board.print()

        while True:
            player = next(alternator)
            move = player.act()
            
            # Make the move
            current_board = current_board.act(move)
            
            # Update both players with new board state
            self.p1.set_board(current_board)
            self.p2.set_board(current_board)
            
            if print_state:
                current_board.print()
            
            # Check if game ended
            result = current_board.isEnd()
            if result != Result.Incomplete:
                return result

def play_game():
    """Play a single game between human and MCTS"""
    human = HumanPlayer()
    mcts_player = MCTSPlayer(iterations=2000)
    judger = Judger(human, mcts_player)
    
    result = judger.play()
    
    if result == Result.X_Wins:
        print("🎉 Congratulations! You win!")
    elif result == Result.O_Wins:
        print("😞 MCTS wins! Better luck next time!")
    else:
        print("🤝 It's a tie! Well played!")
    
    return result

if __name__ == '__main__':
    play_game()