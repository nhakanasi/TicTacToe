import os
import sys

# Add parent directory to path so this file can be run directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from strat.encode import Board, Cell, Result
from strat import config
def outcome_score(result: Result, root_symbol: Cell) -> int:
    if result == Result.Draw:
        return 0
    if (result == Result.X_Wins and root_symbol == Cell.X) or \
       (result == Result.O_Wins and root_symbol == Cell.O):
        return 1
    else:
        return -1

def minimax(board: Board, alpha: float, beta: float, current_symbol: Cell, root_symbol: Cell) -> int:
    result = board.isEnd()
    if result != Result.Incomplete:
        return outcome_score(result, root_symbol)

    maximizing = (current_symbol == root_symbol)
    if maximizing:
        best = -999
        for move in board.get_valid_moves():
            child = board.act(move, current_symbol)
            # Convert symbol properly for next player
            next_symbol = Cell.O if current_symbol == Cell.X else Cell.X
            val = minimax(child, alpha, beta, next_symbol, root_symbol)
            best = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best
    else:
        best = 999
        for move in board.get_valid_moves():
            child = board.act(move, current_symbol)
            # Convert symbol properly for next player
            next_symbol = Cell.O if current_symbol == Cell.X else Cell.X
            val = minimax(child, alpha, beta, next_symbol, root_symbol)
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best

def play_game():
    board = Board()
    human_symbol = Cell.X
    ai_symbol = Cell.O  # Use Cell.O from encode
    current_player = Cell.X  # X starts

    while True:
        board.print()
        if current_player == human_symbol:
            try:
                coord_input = input(f"Input your position (row col) [0-{config.BOARD_ROWS-1}] [0-{config.BOARD_COLS-1}]: ")
                row, col = map(int, coord_input.split())
                if 0 <= row < config.BOARD_ROWS and 0 <= col < config.BOARD_COLS:
                    move = row * config.BOARD_COLS + col
                    if board.isValid(move):
                        board = board.act(move, human_symbol)
                        current_player = ai_symbol
                    else:
                        print("Not empty. Try again.")
                else:
                    print("Invalid coordinates. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Use: row col")
        else:
            print("AI is thinking...")
            best_move = None
            best_value = -999
            for move in board.get_valid_moves():
                child = board.act(move, ai_symbol)
                # Next turn is human's turn
                val = minimax(child, -999, 999, human_symbol, ai_symbol)
                if val > best_value:
                    best_value = val
                    best_move = move

            board = board.act(best_move, ai_symbol)
            ai_r, ai_c = divmod(best_move, config.BOARD_COLS)
            print(f"AI played: {ai_r} {ai_c}")
            current_player = human_symbol

        result = board.isEnd()
        if result != Result.Incomplete:
            board.print()
            if result == Result.X_Wins:
                print("X wins!")
            elif result == Result.O_Wins:
                print("O wins!")
            else:
                print("Draw!")
            break

if __name__ == "__main__":
    play_game()