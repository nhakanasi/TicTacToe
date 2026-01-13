"""Common player interfaces and wrappers for all strategies."""
from __future__ import annotations

import os
import sys
from typing import Protocol, Optional, Callable
import numpy as np

# Add parent directory to path so this file can be run directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from strat.encode import Board, Cell, Result
from strat import config
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

try:
    import sys
    import os
    # Import MCTS shared player
    mcts_shared_path = os.path.join(os.path.dirname(__file__), 'mcts_shared.py')
    if os.path.exists(mcts_shared_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("mcts_shared", mcts_shared_path)
        mcts_shared_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcts_shared_module)
        MCTSSharedPlayer = mcts_shared_module.MCTSPlayer
    else:
        MCTSSharedPlayer = None
except Exception:
    MCTSSharedPlayer = None

try:
    from strat.alpha_zero_large import AlphaGoGomoku
except Exception:
    AlphaGoGomoku = None


class Player(Protocol):
    symbol: Optional[Cell]

    def reset(self) -> None: ...
    def set_board(self, board: Board) -> None: ...
    def set_state(self, board: Board) -> None: ...
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

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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
                if row < 0 or row >= config.BOARD_ROWS or col < 0 or col >= config.BOARD_COLS:
                    print(f"Row/col must be in [0,{config.BOARD_ROWS-1}]x[0,{config.BOARD_COLS-1}]")
                    continue
                move = row * config.BOARD_COLS + col
                if not self.board.isValid(move):
                    print("Cell is occupied. Try again.")
                    continue
                return move
            except ValueError:
                print("Invalid input. Use: row col")


class MCTSAgent:
    def __init__(self, iterations: int = 2000, exploration_constant: float = 1.4) -> None:
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.mcts = MCTS(exploration_constant=exploration_constant)
        self.iterations = iterations

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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
        self.game_state = None

    def reset(self) -> None:
        self.game_state = None

    def set_board(self, board: Board) -> None:
        self.board = board
        if self.game_state is None or self.game_state.board.cells.tobytes() != board.cells.tobytes():
            self.game_state = HeuristicGameState()
            self.game_state.board = board

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

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
                pass

    def act(self) -> Optional[int]:
        assert self.qtable_player is not None, "Q-table player not initialized; setSymbol must be called first"
        assert self.board is not None, "Board not set"
        self.qtable_player.set_state(self.board)
        move, _ = self.qtable_player.act()
        return move


class MCTSSharedAgent:
    """MCTS with shared tree (works best for 3x3, can save/load policy)."""

    def __init__(self, num_simulations: int = 5000, model_path: Optional[str] = None, 
                 use_shared_tree: bool = True, trained_mode: bool = False) -> None:
        if MCTSSharedPlayer is None:
            raise ImportError("MCTS shared module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.num_simulations = num_simulations
        self.model_path = model_path
        self.use_shared_tree = use_shared_tree
        self.trained_mode = trained_mode
        self.mcts_player = None

    def reset(self) -> None:
        if self.mcts_player:
            self.mcts_player.reset()

    def set_board(self, board: Board) -> None:
        self.board = board
        if self.mcts_player:
            self.mcts_player.set_state(board)

    def set_state(self, board: Board) -> None:
        # Compatibility: Judger expects set_state
        self.set_board(board)

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol
        # Initialize MCTS player
        self.mcts_player = MCTSSharedPlayer(
            num_sim=self.num_simulations, 
            use_shared_tree=self.use_shared_tree,
            trained_mode=self.trained_mode
        )
        self.mcts_player.setSymbol(symbol)
        # Load policy if path provided
        if self.model_path:
            try:
                self.mcts_player.load_policy()
            except Exception:
                pass

    def act(self) -> Optional[int]:
        assert self.mcts_player is not None, "MCTS player not initialized; setSymbol must be called first"
        assert self.board is not None, "Board not set"
        self.mcts_player.set_state(self.board)
        move = self.mcts_player.act(temperature=0.0, training=False)
        return move


class NeuralMCTSAgent:
    """AlphaGo-style neural MCTS agent (for trained models on larger boards)."""

    def __init__(self, model_path: Optional[str] = None, num_simulations: int = 800) -> None:
        if AlphaGoGomoku is None:
            raise ImportError("AlphaGo module not available")
        self.symbol: Optional[Cell] = None
        self.board: Optional[Board] = None
        self.model_path = model_path
        self.num_simulations = num_simulations
        self.agent = None
        
        # Eagerly initialize and validate model works with current board size
        if self.model_path:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.agent = AlphaGoGomoku(model_path=self.model_path, device=device)
            # Test forward pass to validate model matches current board size
            test_board = Board()
            self.agent.get_policy_and_value(test_board)

    def reset(self) -> None:
        pass

    def set_board(self, board: Board) -> None:
        self.board = board

    def setSymbol(self, symbol: Cell) -> None:
        self.symbol = symbol

    def act(self) -> Optional[int]:
        assert self.agent is not None, "Neural MCTS not initialized"
        assert self.board is not None, "Board not set"
        move, _, _ = self.agent.search(self.board, num_simulations=self.num_simulations, temperature=0.0)
        return move
