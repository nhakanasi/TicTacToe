"""
Gomoku AI with Negamax Algorithm and Alpha-Beta Pruning
Uses existing Board class from board.py with custom encoder
"""

import numpy as np
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from strat.config import BOARD_ROWS, BOARD_COLS, WIN, BOARD_SIZE
from strat.encode import Board, Cell, Result


# Score Constants
EVAL_WIN_BASE = 1_000_000_000
EVAL_MIN = -EVAL_WIN_BASE - 100
EVAL_MAX = EVAL_WIN_BASE + 100

# Heuristic Scores
SCORE_OPEN_4 = 100_000_000
SCORE_CLOSED_4 = 5_000_000
SCORE_OPEN_3 = 1_000_000
SCORE_CLOSED_3 = 10_000
SCORE_OPEN_2 = 5_000
SCORE_CLOSED_2 = 100
SCORE_OPEN_1 = 10


@dataclass
class PlayRange:
    """Tracks the active play area for optimization"""
    min_r: int
    max_r: int
    min_c: int
    max_c: int
    
    def update(self, row: int, col: int) -> None:
        """Expand range to include new move with 2-cell margin"""
        self.min_r = max(0, min(self.min_r, row - 2))
        self.max_r = min(BOARD_ROWS - 1, max(self.max_r, row + 2))
        self.min_c = max(0, min(self.min_c, col - 2))
        self.max_c = min(BOARD_COLS - 1, max(self.max_c, col + 2))


class BoardEncoder:
    """Custom board encoder for neural network input or state representation"""
    
    @staticmethod
    def encode_board(board: Board, current_player: Cell = None) -> np.ndarray:
        """
        Encode board state into multi-channel representation
        Returns: 4-channel array (BOARD_ROWS, BOARD_COLS, 4)
        - Channel 0: Current player pieces (1 where player has piece, 0 otherwise)
        - Channel 1: Opponent pieces (1 where opponent has piece, 0 otherwise)
        - Channel 2: Empty cells (1 where empty, 0 otherwise)
        - Channel 3: Current player indicator (all 1s if X, all 0s if O)
        """
        encoded = np.zeros((BOARD_ROWS, BOARD_COLS, 4), dtype=np.float32)
        
        cells_2d = board.visualize()
        
        # Determine current player from move count if not specified
        if current_player is None:
            num_moves = np.count_nonzero(board.cells)
            current_player = Cell.X if num_moves % 2 == 0 else Cell.O
        
        # Channel 0: Current player's pieces
        encoded[:, :, 0] = (cells_2d == current_player).astype(np.float32)
        
        # Channel 1: Opponent's pieces
        opponent = Cell.O if current_player == Cell.X else Cell.X
        encoded[:, :, 1] = (cells_2d == opponent).astype(np.float32)
        
        # Channel 2: Empty cells
        encoded[:, :, 2] = (cells_2d == Cell.Empty).astype(np.float32)
        
        # Channel 3: Current player indicator
        encoded[:, :, 3] = 1.0 if current_player == Cell.X else 0.0
        
        return encoded
    
    @staticmethod
    def encode_1d(board: Board, current_player: Cell = None) -> np.ndarray:
        """
        Simple 1D encoding from current player's perspective
        Returns: Flattened array where 1=current player, -1=opponent, 0=empty
        """
        cells = board.cells.copy()
        
        if current_player is None:
            num_moves = np.count_nonzero(board.cells)
            current_player = Cell.X if num_moves % 2 == 0 else Cell.O
        
        if current_player == Cell.O:
            cells = -cells  # Flip perspective
        return cells
    
    @staticmethod
    def get_symmetries(board: Board) -> List[np.ndarray]:
        """
        Generate all 8 symmetric transformations of the board
        Returns list of board states
        """
        symmetries = []
        cells_2d = board.visualize()
        
        # 4 rotations
        for k in range(4):
            rotated = np.rot90(cells_2d, k)
            symmetries.append(rotated.flatten())
        
        # Mirror + 4 rotations
        mirrored = np.fliplr(cells_2d)
        for k in range(4):
            rotated = np.rot90(mirrored, k)
            symmetries.append(rotated.flatten())
        
        return symmetries


class GameState:
    """Wrapper for Board with move history and play range optimization"""
    
    def __init__(self):
        self.board = Board()
        self.move_history = []
        self.play_ranges = [PlayRange(
            max(0, BOARD_ROWS // 2 - 2),
            min(BOARD_ROWS - 1, BOARD_ROWS // 2 + 2),
            max(0, BOARD_COLS // 2 - 2),
            min(BOARD_COLS - 1, BOARD_COLS // 2 + 2)
        )]  # Start center-ish
    
    def _index_to_coords(self, idx: int) -> Tuple[int, int]:
        """Convert 1D index to 2D coordinates"""
        return idx // BOARD_COLS, idx % BOARD_COLS
    
    def get_current_player(self) -> Cell:
        """Determine current player from move count"""
        num_moves = len(self.move_history)
        return Cell.X if num_moves % 2 == 0 else Cell.O
    
    def make_move(self, move: int) -> None:
        """Make a move and update play range"""
        symbol = self.get_current_player()
        self.board = self.board.act(move, symbol)
        self.move_history.append(move)
        
        # Update play range
        row, col = self._index_to_coords(move)
        new_range = PlayRange(
            self.play_ranges[-1].min_r,
            self.play_ranges[-1].max_r,
            self.play_ranges[-1].min_c,
            self.play_ranges[-1].max_c
        )
        new_range.update(row, col)
        self.play_ranges.append(new_range)
    
    def undo_move(self) -> None:
        """Undo the last move"""
        if not self.move_history:
            return
        
        move = self.move_history.pop()
        self.play_ranges.pop()
        
        # Reconstruct board from history
        self.board = Board()
        for m in self.move_history:
            num_moves = np.count_nonzero(self.board.cells)
            symbol = Cell.X if num_moves % 2 == 0 else Cell.O
            self.board = self.board.act(m, symbol)
    
    def get_ordered_moves(self) -> List[int]:
        """Get legal moves ordered by likelihood (moves near existing pieces first)"""
        valid_moves = self.board.get_valid_moves()
        
        if not self.move_history:
            return valid_moves
        
        play_range = self.play_ranges[-1]
        
        # Filter moves within play range
        range_moves = []
        for move in valid_moves:
            r, c = self._index_to_coords(move)
            if (play_range.min_r <= r <= play_range.max_r and 
                play_range.min_c <= c <= play_range.max_c):
                range_moves.append(move)
        
        # Sort: moves with neighbors first
        def has_neighbor(idx: int) -> bool:
            r, c = self._index_to_coords(idx)
            board_2d = self.board.visualize()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                        if board_2d[nr, nc] != Cell.Empty:
                            return True
            return False
        
        range_moves.sort(key=lambda m: not has_neighbor(m))
        return range_moves if range_moves else valid_moves


def is_valid_pos(r: int, c: int) -> bool:
    """Check if position is within board bounds"""
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS


def evaluate(game_state: GameState) -> int:
    """
    Evaluate board position from current player's perspective
    Higher score means better for current player
    """
    score_x = 0
    score_o = 0
    
    board_2d = game_state.board.visualize()
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            cell_type = board_2d[r, c]
            if cell_type == Cell.Empty:
                continue
            
            for dr, dc in directions:
                # Skip if previous cell in this direction has same type (avoid double counting)
                prev_r, prev_c = r - dr, c - dc
                if is_valid_pos(prev_r, prev_c) and board_2d[prev_r, prev_c] == cell_type:
                    continue
                
                # Count consecutive pieces
                count = 0
                curr_r, curr_c = r, c
                while is_valid_pos(curr_r, curr_c) and board_2d[curr_r, curr_c] == cell_type:
                    count += 1
                    curr_r += dr
                    curr_c += dc
                
                # Count open ends
                open_ends = 0
                if is_valid_pos(prev_r, prev_c) and board_2d[prev_r, prev_c] == Cell.Empty:
                    open_ends += 1
                if is_valid_pos(curr_r, curr_c) and board_2d[curr_r, curr_c] == Cell.Empty:
                    open_ends += 1
                
                # Score this pattern
                current_score = 0
                if count >= WIN:
                    current_score = EVAL_WIN_BASE
                elif count == WIN - 1:
                    current_score = SCORE_OPEN_4 if open_ends == 2 else (SCORE_CLOSED_4 if open_ends == 1 else 0)
                elif count == WIN - 2:
                    current_score = SCORE_OPEN_3 if open_ends == 2 else (SCORE_CLOSED_3 if open_ends == 1 else 0)
                elif count == WIN - 3:
                    current_score = SCORE_OPEN_2 if open_ends == 2 else (SCORE_CLOSED_2 if open_ends == 1 else 0)
                elif count == 1 and open_ends == 2:
                    current_score = SCORE_OPEN_1
                
                if cell_type == Cell.X:
                    score_x += current_score
                else:
                    score_o += current_score
    
    # Return from current player's perspective
    current_player = game_state.get_current_player()
    if current_player == Cell.X:
        return score_x - score_o
    return score_o - score_x


def negamax(game_state: GameState, depth: int, alpha: int, beta: int) -> int:
    """
    Negamax algorithm with alpha-beta pruning
    Returns evaluation score from current player's perspective
    """
    # Check for winner
    result = game_state.board.isEnd()
    if result != Result.Incomplete:
        if result == Result.Draw:
            return 0
        # Current player has lost (opponent just won)
        return -(EVAL_WIN_BASE + depth)
    
    # Terminal depth
    if depth == 0:
        return evaluate(game_state)
    
    moves = game_state.get_ordered_moves()
    if not moves:
        return 0  # Draw
    
    max_eval = EVAL_MIN
    for move in moves:
        game_state.make_move(move)
        eval_score = -negamax(game_state, depth - 1, -beta, -alpha)
        game_state.undo_move()
        
        max_eval = max(max_eval, eval_score)
        alpha = max(alpha, eval_score)
        
        if alpha >= beta:
            break  # Beta cutoff
    
    return max_eval


def get_best_move(game_state: GameState, depth: int = 4) -> Tuple[int, int]:
    """
    Find the best move using negamax with alpha-beta pruning
    Returns: (move_index, evaluation_score)
    """
    moves = game_state.get_ordered_moves()
    if not moves:
        return BOARD_SIZE // 2, 0
    
    best_move = moves[0]
    max_eval = EVAL_MIN
    alpha = EVAL_MIN
    beta = EVAL_MAX
    
    print(f"Thinking (Depth {depth})... ", end="", flush=True)
    start_time = time.time()
    
    for move in moves:
        game_state.make_move(move)
        eval_score = -negamax(game_state, depth - 1, -beta, -alpha)
        game_state.undo_move()
        
        # Found a forced win?
        if eval_score >= EVAL_WIN_BASE:
            print(f"Winning move found! (Score: {eval_score})")
            best_move = move
            max_eval = eval_score
            break
        
        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move
        
        alpha = max(alpha, eval_score)
    
    elapsed = (time.time() - start_time) * 1000
    print(f"Done in {elapsed:.1f}ms. (Eval: {max_eval})")
    
    return best_move, max_eval


def play_game(depth: int = 4):
    """Main game loop for human vs AI"""
    game_state = GameState()
    print(f"Gomoku AI (Depth: {depth}, Board: {BOARD_ROWS}x{BOARD_COLS}, Win: {WIN})")
    print("You are 'X'. Enter moves as 'row col' or single index.")
    print()
    
    while True:
        game_state.board.print()
        print()
        
        result = game_state.board.isEnd()
        if result != Result.Incomplete:
            if result == Result.X_Wins:
                print("Winner: Player (X)")
            elif result == Result.O_Wins:
                print("Winner: AI (O)")
            else:
                print("Draw!")
            break
        
        current_player = game_state.get_current_player()
        
        if current_player == Cell.X:
            # Human player
            while True:
                try:
                    user_input = input("Your move: ").strip()
                    parts = user_input.split()
                    
                    if len(parts) == 1:
                        # Single index input
                        move = int(parts[0])
                    elif len(parts) == 2:
                        # Row col input
                        r, c = map(int, parts)
                        move = r * BOARD_COLS + c
                    else:
                        print("Invalid input. Enter 'row col' or index.")
                        continue
                    
                    if move in game_state.board.get_valid_moves():
                        game_state.make_move(move)
                        break
                    else:
                        print("Invalid move. Try again.")
                except (ValueError, IndexError):
                    print("Invalid input. Enter 'row col' or index.")
        else:
            # AI player
            move, eval_score = get_best_move(game_state, depth)
            row = move // BOARD_COLS
            col = move % BOARD_COLS
            print(f"AI plays: {row} {col} (index: {move})")
            game_state.make_move(move)
        
        print()