import numpy as np
import os
import sys
from math import sqrt, log

# Add parent directory to path so this file can be run directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from strat.encode import Board, Cell, Result
from strat import config


class MCTS:
    class Node:
        def __init__(self, board, canonical_transform=0, exploration_constant=1.4):
            self.board = board
            self.canonical_transform = canonical_transform
            self.exploration_constant = exploration_constant

            self.wins = 0
            self.draws = 0
            self.losses = 0
            self.visits = 0

            self.children = {}   # move -> Node
            self.parent = None
            self.parent_move = None

        def upperBound(self, parent_visits):
            if self.visits == 0:
                return 1e8
            exploitation = (self.wins + 0.5 * self.draws) / self.visits
            exploration = self.exploration_constant * sqrt(log(parent_visits) / self.visits)
            return exploitation + exploration

        def isFullyExpanded(self):
            valid_moves = self.board.get_valid_moves()
            return len(self.children) == len(valid_moves)

    def __init__(self, max_nodes=500000, exploration_constant=1.4):
        self.nodes = {}
        self.MAX_NODES = max_nodes
        self.exploration_constant = exploration_constant

    # -----------------------------
    # NODE DEDUP USING CANONICAL KEY
    # -----------------------------
    def get_createNode(self, board):
        canonical_cells, transform_id = board.get_canonical()
        key = str(canonical_cells)

        if key not in self.nodes:
            if len(self.nodes) >= self.MAX_NODES:
                return MCTS.Node(board, transform_id, self.exploration_constant)  # fallback no store
            self.nodes[key] = MCTS.Node(board, transform_id, self.exploration_constant)

        return self.nodes[key]

    # -----------------------------
    # SELECTION
    # -----------------------------
    def select(self, node):
        best_score = -1
        best_child = None
        best_move = None

        for move, child in node.children.items():
            score = child.upperBound(node.visits)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    # -----------------------------
    # EXPANSION WITH TRANSFORM MAP BACK
    # -----------------------------
    def expand(self, node):
        board = node.board
        valid = board.get_valid_moves()
        unexplored = [m for m in valid if m not in node.children]

        if not unexplored:  # Safety check
            return node
            
        move = unexplored[np.random.randint(len(unexplored))]

        new_board = board.act(move)
        child = self.get_createNode(new_board)
        child.parent = node
        child.parent_move = move

        node.children[move] = child
        return child

    # -----------------------------
    # RANDOM SIMULATION
    # -----------------------------
    def simulate(self, board):
        curr = board.clone()

        while True:
            r = curr.isEnd()
            if r != Result.Incomplete:
                return r

            moves = curr.get_valid_moves()
            m = moves[np.random.randint(len(moves))]
            curr = curr.act(m)

    # -----------------------------
    # BACKPROP
    # -----------------------------
    def backprop(self, node, result):
        while node is not None:
            node.visits += 1

            depth = node.board.get_depth()
            node_player = Cell.X if depth % 2 == 0 else Cell.O

            if result == Result.Draw:
                node.draws += 1
            elif (result == Result.X_Wins and node_player == Cell.X) or \
                 (result == Result.O_Wins and node_player == Cell.O):
                node.wins += 1
            else:
                node.losses += 1

            node = node.parent

    # -----------------------------
    # RUN MCTS ITERATIONS
    # -----------------------------
    def search(self, board, iters=2000):
        root = self.get_createNode(board)

        for _ in range(iters):
            node = root
            b = board.clone()

            # SELECTION
            while node.children and node.isFullyExpanded():
                move, node = self.select(node)
                b = b.act(move)

            # EXPANSION (only if game is not over)
            if b.isEnd() == Result.Incomplete:
                node = self.expand(node)
                b = node.board

            # SIMULATION
            result = self.simulate(b)

            # BACKPROP
            self.backprop(node, result)

        best_child = None
        best_visits = -1
        best_move = None

        for move, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child
                best_move = move

        if root.canonical_transform != 0:
            best_move = Board.inverse_transform_move_1d(best_move, root.canonical_transform)

        return best_move