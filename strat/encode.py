import numpy as np
import pickle
import os
import typing
from math import sqrt, log2 as log
from enum import IntEnum
BOARD_ROWS = int(input("Rows: "))
BOARD_COLS = int(input("Cols: "))
WIN = int(input("Wins: "))
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
PATH = f"Reinforecedment learning/tictactoe/policy/board_{BOARD_ROWS}x{BOARD_COLS}"
C = sqrt(2)
os.makedirs(PATH, exist_ok=True)

class Cell(IntEnum):
    Empty = 0 
    O = -1
    X = +1

class Result(IntEnum):
    X_Wins = 1
    O_Wins = -1
    Draw = 0
    Incomplete = 2

class Board: # State?
    def __init__(self, cells=None):
        self.winner = None
        self._hash_val = None
        self._canonical_cache = None  # Cache for canonical form
        if cells is None:
            self.cells = np.array([Cell.Empty] * BOARD_SIZE)
        else:
            self.cells = cells.copy()

    @staticmethod
    def all_transforms(cells_1d):
        """Generate all 8 symmetry transforms of a 1D board array."""
        xs = []
        board_2d = cells_1d.reshape(BOARD_ROWS, BOARD_COLS)
        
        # 4 rotations
        xs.append(board_2d.flatten())
        xs.append(np.rot90(board_2d, 1).flatten())
        xs.append(np.rot90(board_2d, 2).flatten())
        xs.append(np.rot90(board_2d, 3).flatten())

        # mirror + 4 rotations
        m = np.flip(board_2d, axis=1)
        xs.append(m.flatten())
        xs.append(np.rot90(m, 1).flatten())
        xs.append(np.rot90(m, 2).flatten())
        xs.append(np.rot90(m, 3).flatten())

        return xs

    @staticmethod
    def transform_move_1d(move_1d, transform):
        """Apply transform to a 1D move index. transform ∈ [0..7]."""
        # Convert 1D to 2D
        r, c = divmod(move_1d, BOARD_COLS)
        
        if transform == 0:        # R0
            new_r, new_c = r, c
        elif transform == 1:      # R90
            new_r, new_c = c, BOARD_ROWS-1-r
        elif transform == 2:      # R180
            new_r, new_c = BOARD_ROWS-1-r, BOARD_COLS-1-c
        elif transform == 3:      # R270
            new_r, new_c = BOARD_COLS-1-c, r
        else:
            # Mirror horizontally first: (r, c) -> (r, BOARD_COLS-1-c)
            c = BOARD_COLS-1-c
            if transform == 4:    # Mirror
                new_r, new_c = r, c
            elif transform == 5:  # Mirror + R90
                new_r, new_c = c, BOARD_ROWS-1-r
            elif transform == 6:  # Mirror + R180
                new_r, new_c = BOARD_ROWS-1-r, BOARD_COLS-1-c
            elif transform == 7:  # Mirror + R270
                new_r, new_c = BOARD_COLS-1-c, r
        
        # Convert back to 1D
        return new_r * BOARD_COLS + new_c

    @staticmethod
    def inverse_transform_move_1d(move_1d, transform):
        """Invert the symmetry transform of a 1D move."""
        inv = [0, 3, 2, 1, 4, 7, 6, 5]
        return Board.transform_move_1d(move_1d, inv[transform])

    def get_canonical(self):
        """Return (canonical_cells, transform_id) that maps original to canonical."""
        if self._canonical_cache is not None:
            return self._canonical_cache
            
        transforms = self.all_transforms(self.cells)
        strings = [str(x) for x in transforms]
        transform_id = np.argmin(strings)
        
        canonical_cells = transforms[transform_id]
        self._canonical_cache = (canonical_cells, transform_id)
        return self._canonical_cache

    def get_canonical_hash(self):
        """Get hash based on canonical form for deduplication."""
        canonical_cells, _ = self.get_canonical()
        return hash(tuple(canonical_cells))

    def hash(self):
        if self._hash_val is None:
            self._hash_val = hash(tuple(self.cells))
        return self._hash_val
    
    def clone(self):
        """Create a deep copy of the board."""
        new_board = Board(self.cells)
        new_board.winner = self.winner
        return new_board

    def next_player(self):
        """Get the next player to move."""
        num_moves = np.count_nonzero(self.cells)
        return Cell.X if num_moves % 2 == 0 else Cell.O
    
    def visualize(self):
        return self.cells.reshape(BOARD_ROWS, BOARD_COLS)

    def execTurn(self, index, symbol):
        self.cells[index] = symbol
        self._hash_val = None  # Invalidate hash cache
        self._canonical_cache = None  # Invalidate canonical cache
        
    def alternate(self):
        while True:
            yield Cell.X
            yield Cell.O
    
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
        new_board._canonical_cache = None
        
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

    def isValid(self, iloc):
        if iloc >= (BOARD_SIZE) or iloc < 0:
            return False

        if self.cells[iloc] == Cell.Empty:
            return True

        return False
    
    def get_valid_moves(self):
        return [i for i in range(BOARD_SIZE)
                if self.cells[i] == Cell.Empty]
    
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
            self.winner = "X wins"
            return Result.X_Wins

        if min_value == -WIN:
            self.winner = "O wins"
            return Result.O_Wins

        if not self.get_valid_moves():
            self.winner = "Draw"
            return Result.Draw

        return Result.Incomplete
    
    def get_depth(self):
        return sum(cell != Cell.Empty for cell in self.cells)