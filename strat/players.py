"""Common player interfaces and wrappers for all strategies."""
from __future__ import annotations

from typing import Protocol, Optional, Callable
import numpy as np

from strat.encode import Board, Cell, Result, BOARD_ROWS, BOARD_COLS
from strat.mcts import MCTS

try:
    from strat.train_heuristic import TDPlayer
except Exception:
    TDPlayer = None

try:
    from strat.alpha_beta import minimax as ab_minimax
except Exception:
    ab_minimax = None

try:
    from strat.minimax import minimax as mm_minimax
except Exception:
    mm_minimax = None

try:
    from strat.heuristic import GameState as HeuristicGameState, get_best_move as heuristic_best_move
except Exception:
    HeuristicGameState = None
    heuristic_best_move = None

try:
    import sys
    import os
    # Import Q-table player (note: only works for small boards like 3x3)
    qtable_path = os.path.join(os.path.dirname(__file__), 'q table.py')
    if os.path.exists(qtable_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("qtable", qtable_path)
        qtable_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qtable_module)
        QTablePlayer = qtable_module.Player
    else:
        QTablePlayer = None
except Exception:
    QTablePlayer = None


class Player(Protocol):
    symbol: Optional[Cell]

    def reset(self) -> None: ...
    def set_board(self, board: Board) -> None: ...
    def setSymbol(self, symbol: Cell) -> None: ...
    def act(self) -> Optional[int]: ...


class HumanCLI:
    """Human player reading moves from stdin."""

    def __init__(self) -> None:
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol

    def act(self) -> Optional[int]:
        assert self.board is not None, "Board not set"
        self.board.print()
        while True:
            try:
                user_input = input("Enter your move (row col): ").strip()
                if user_input.lower() in {"quit", "q", "exit"}:
                    return None
                parts = user_input.split()
                if len(parts) != 2:
                    print("Enter row and col separated by space.")
                    continue
                row, col = int(parts[0]), int(parts[1])
                if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
                    print(f"Row/col must be in [0,{BOARD_ROWS-1}]x[0,{BOARD_COLS-1}]")
                    continue
                move = row * BOARD_COLS + col
                if not self.board.isValid(move):
                    print("Cell is occupied. Try again.")
                    continue
                return move
            except ValueError:
                print("Invalid input. Use: row col")


class MCTSAgent:
    def __init__(self, iterations: int = 2000) -> None:
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.mcts = MCTS()
        self.iterations = iterations

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol

    def act(self) -> Optional[int]:
        assert self.board is not None, "Board not set"
        move = self.mcts.search(self.board, self.iterations)
        return move


class TDPolicy:
    """Wrapper around train_heuristic.TDPlayer (trainable eval)."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if TDPlayer is None:
            raise ImportError("train_heuristic module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.model_path = model_path

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol
        if self.model_path is None:
            raise ValueError("TDPolicy requires a model path to load weights")
        # Recreate underlying player when symbol changes
        self.td = TDPlayer(self.model_path, symbol)

    def act(self) -> Optional[int]:
        assert self.board is not None, "Board not set"
        assert self.td is not None, "TD model not initialized; setSymbol must be called first"
        return self.td.get_move(self.board)


class MinimaxAgent:
    """Pure minimax (no alpha-beta)."""

    def __init__(self) -> None:
        if mm_minimax is None:
            raise ImportError("minimax module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol

    def act(self) -> Optional[int]:
        assert self.board is not None and self.symbol is not None
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            return None
        best_move = None
        best_val = -999_999
        opponent = Cell.O if self.symbol == Cell.X else Cell.X
        for mv in valid_moves:
            child = self.board.act(mv, self.symbol)
            val = mm_minimax(child, opponent, self.symbol)
            if val > best_val:
                best_val = val
                best_move = mv
        return best_move


class AlphaBetaAgent:
    """Alpha-beta minimax agent."""

    def __init__(self, depth: int = 6) -> None:
        if ab_minimax is None:
            raise ImportError("alpha_beta module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.depth = depth

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol

    def act(self) -> Optional[int]:
        assert self.board is not None and self.symbol is not None
        valid_moves = self.board.get_valid_moves()
        if not valid_moves:
            return None
        best_move = None
        best_val = -999_999
        opponent = Cell.O if self.symbol == Cell.X else Cell.X
        for mv in valid_moves:
            child = self.board.act(mv, self.symbol)
            val = ab_minimax(child, -999_999, 999_999, opponent, self.symbol)
            if val > best_val:
                best_val = val
                best_move = mv
        return best_move


class HeuristicAgent:
    def __init__(self, depth: int = 4) -> None:
        if HeuristicGameState is None or heuristic_best_move is None:
            raise ImportError("heuristic module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.depth = depth

    def reset(self) -> None:
        self.game_state = None

    def set_board(self, board: Board) -> None:
        self.board = board
        if self.game_state is None or self.game_state.board.cells.tobytes() != board.cells.tobytes():
            self.game_state = HeuristicGameState()
            self.game_state.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol

    def act(self) -> Optional[int]:
        assert self.game_state is not None
        move, _ = heuristic_best_move(self.game_state, self.depth)
        return move


class QTableAgent:
    """Q-learning table-based agent (works best for small boards like 3x3)."""

    def __init__(self, model_path: Optional[str] = None, epsilon: float = 0.0) -> None:
        if QTablePlayer is None:
            raise ImportError("Q-table module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.model_path = model_path
        self.epsilon = epsilon
        self.qtable_player = None

    def reset(self) -> None:
        if self.qtable_player:
            self.qtable_player.reset()

    def set_board(self, board: Board) -> None:
        self.board = board
        if self.qtable_player:
            self.qtable_player.set_state(board)

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol
        # Initialize Q-table player
        self.qtable_player = QTablePlayer(epsilon=self.epsilon)
        self.qtable_player.setSymbol(symbol)
        # Load policy if path provided
        if self.model_path:
            try:
                import pickle
                with open(self.model_path, 'rb') as f:
                    self.qtable_player.estimations = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Q-table model not found at {self.model_path}, using random policy")

    def act(self) -> Optional[int]:
        assert self.qtable_player is not None, "Q-table player not initialized; setSymbol must be called first"
        assert self.board is not None, "Board not set"
        self.qtable_player.set_state(self.board)
        move, _ = self.qtable_player.act()
        return move
