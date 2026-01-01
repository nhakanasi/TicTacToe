import os
from enum import IntEnum
import numpy as np

from strat.config import BOARD_ROWS, BOARD_COLS, WIN

BOARD_SIZE = BOARD_ROWS * BOARD_COLS
PATH = f"Reinforecedment learning/tictactoe/policy/board_{BOARD_ROWS}x{BOARD_COLS}"
C = 1.0  # Reduced from sqrt(2) for more exploitation
os.makedirs(PATH, exist_ok=True)

class Cell(IntEnum):
    Empty = 0 
    O = 1
    X = -1

class Result(IntEnum):
    X_Wins = 1
    O_Wins = -1
    Draw = 0
    Incomplete = 2

class Board:
    def __init__(self, cells=None):
        self.winner = None
        self._hash_val = None
        if cells is None:
            self.cells = np.array([Cell.Empty] * BOARD_SIZE)
        else:
            self.cells = cells.copy()

    def hash(self, current_symbol=None):
        player_flag = 0
        if current_symbol == Cell.X:
            player_flag = 1
        elif current_symbol == Cell.O:
            player_flag = 2
        return hash((tuple(self.cells), player_flag))
    
    def visualize(self):
        return self.cells.reshape(BOARD_ROWS, BOARD_COLS)

    def act(self, move, symbol=None):
        if symbol is None:
            num_moves = np.count_nonzero(self.cells)
            symbol = Cell.X if num_moves % 2 == 0 else Cell.O
        new_cells = self.cells.copy()
        new_cells[move] = symbol
        new_board = object.__new__(Board)
        new_board.cells = new_cells
        new_board.winner = None
        new_board._hash_val = None
        return new_board

    def print(self):
        print('\n')
        for row in range(BOARD_ROWS):
            print('|', end="")
            for col in range(BOARD_COLS):
                cell = self.visualize()[row][col]
                print("{}".format(["X" if cell == Cell.X else "O" if cell == Cell.O else "."]), end="|")
            if row < BOARD_ROWS - 1:
                print("\n-------------")
        print('\n')

    def get_valid_moves(self):
        return [i for i in range(BOARD_SIZE) if self.cells[i] == Cell.Empty]

    def get_rows_cols_and_diagonals(self):
        board_2d = self.visualize()
        sequences = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS - WIN + 1):
                sequences.append(sum(board_2d[i, j:j+WIN]))
        for j in range(BOARD_COLS):
            for i in range(BOARD_ROWS - WIN + 1):
                sequences.append(sum(board_2d[i:i+WIN, j]))
        for i in range(BOARD_ROWS - WIN + 1):
            for j in range(BOARD_COLS - WIN + 1):
                sequences.append(sum(board_2d[i+k, j+k] for k in range(WIN)))
        for i in range(BOARD_ROWS - WIN + 1):
            for j in range(WIN - 1, BOARD_COLS):
                sequences.append(sum(board_2d[i+k, j-k] for k in range(WIN)))
        return sequences

    def isEnd(self):
        sequences = self.get_rows_cols_and_diagonals()
        max_value = max(sequences)
        min_value = min(sequences)
        if max_value == WIN:
            self.winner = "O wins"
            return Result.O_Wins
        if min_value == -WIN:
            self.winner = "X wins"
            return Result.X_Wins
        if not self.get_valid_moves():
            self.winner = "Draw"
            return Result.Draw
        return Result.Incomplete