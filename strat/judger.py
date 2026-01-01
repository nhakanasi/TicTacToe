"""Shared game referee/loop."""
from __future__ import annotations

from strat.players import Player
from strat.encode import Board, Result, Cell


class Judger:
    def __init__(self, p1: Player, p2: Player):
        self.p1 = p1
        self.p2 = p2

    def reset(self) -> None:
        self.p1.reset()
        self.p2.reset()

    def play(self, print_state: bool = True) -> Result:
        board = Board()
        self.p1.setSymbol(Cell.X)
        self.p2.setSymbol(Cell.O)
        self.p1.set_board(board)
        self.p2.set_board(board)
        turn = [self.p1, self.p2]
        idx = 0

        if print_state:
            print("Game start: X vs O")
            board.print()

        while True:
            player = turn[idx % 2]
            move = player.act()
            if move is None:
                return Result.Draw

            board = board.act(move, player.symbol)
            self.p1.set_board(board)
            self.p2.set_board(board)

            if print_state:
                board.print()

            result = board.isEnd()
            if result != Result.Incomplete:
                return result

            idx += 1
