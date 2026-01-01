# TicTacToe / Gomoku - Exploration of many stratergies

A project about the many ways we can train an agent to play Tic-tac-toe or Gomoku. Some perform better than others.
Checkpoints/Policy are not included

## Quick Start

### Play a Game

```bash
# Human vs MCTS on 3x3 board
python play.py --rows 3 --cols 3 --win 3 --player human-vs-mcts

# Human vs Heuristic on 10x10 Gomoku board
python play.py --rows 10 --cols 10 --win 5 --player human-vs-heuristic

# Watch AI vs AI: MCTS vs AlphaBeta
python play.py --rows 5 --cols 5 --win 4 --player mcts-vs-alphabeta
```

### Train an Agent

```bash
# Train TD agent for 10x10 Gomoku
python train.py --agent td --rows 10 --cols 10 --win 5 --episodes 10000

# Train AlphaGo-style neural MCTS
python train_gomoku_alphago.py --rows 10 --cols 10 --win 5 --iterations 20
```

## Features

- **Multiple AI Agents**: MCTS, Minimax, Alpha-Beta, Heuristic, Q-Table, TD Learning, Neural MCTS
- **Flexible Board Sizes**: Any grid size (e.g., 3x3, 5x5, 10x10, 15x15)
- **Customizable Win Conditions**: 3-in-a-row, 4-in-a-row, 5-in-a-row (Gomoku), etc.
- **Any Matchup**: Human vs AI, AI vs AI, or even Human vs Human
- **Training Support**: Train Q-Table, TD agents and neural network-based MCTS agents
- **Checkpoint System**: Save and resume training progress

## Available Agents

| Agent | Description | Strength | Training Required | Best For |
|-------|-------------|----------|-------------------|----------|
| `human` | Human player via CLI | N/A | No | - |
| `mcts` | Monte Carlo Tree Search | Strong | No | All sizes |
| `alphabeta` | Alpha-beta pruning minimax | Medium-Strong | No | Small-Medium |
| `minimax` | Pure minimax search | Medium | No | Small boards |
| `heuristic` | Negamax with heuristics | Medium | No | Medium-Large |
| `qtable` | Q-learning table-based | Varies | Yes | Small (3x3) |
| `td` | Temporal Difference learning | Varies | Yes | Any size |
| `Neural MCTS`| AlphaGo-style neural network | Not trained well yet | Yes | Large boards |

## Requirements

- Python 3.7+
- PyTorch (for neural network agents)
- NumPy

## Install dependencies:
```bash
pip install torch numpy
```

## Afterwords

![alt text](image.png)