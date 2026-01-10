"""Configuration for board dimensions and win condition.
Defaults can be overridden via CLI before creating boards.
"""

BOARD_ROWS = 3
BOARD_COLS = 3
WIN = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


def set_board(rows: int, cols: int, win: int) -> None:
    """Set global board dimensions and win length."""
    global BOARD_ROWS, BOARD_COLS, WIN, BOARD_SIZE
    BOARD_ROWS = int(rows)
    BOARD_COLS = int(cols)
    WIN = int(win)
    BOARD_SIZE = BOARD_ROWS * BOARD_COLS
