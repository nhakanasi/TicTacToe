import os
import numpy as np
import typing
import pickle
from strat.encode import Board, Cell, Result
from strat import config

def getAllStateImpl(currentboard: Board, currentsymbol: Cell, allstates):
    for move in currentboard.get_valid_moves():
        newboard = currentboard.act(move, currentsymbol)
        newhash = newboard.hash()
        if newhash not in allstates:
            result = newboard.isEnd()
            isend = (result != Result.Incomplete)
            allstates[newhash] = (newboard, isend)
            if not isend:
                next_symbol = Cell.O if currentsymbol == Cell.X else Cell.X
                getAllStateImpl(newboard, next_symbol, allstates)

def getAllState():
    currentsymbol = Cell.X
    currentboard = Board()
    allstate = dict()
    allstate[currentboard.hash()] = (currentboard, currentboard.isEnd() != Result.Incomplete)
    getAllStateImpl(currentboard, currentsymbol, allstate)
    return allstate

allstate = getAllState() # Generate all row value (States for Q table Rows)

class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.currentplayer = None
        self.p1_symbol = Cell.X  # 1
        self.p2_symbol = Cell.O  # -1
        self.p1.setSymbol(self.p1_symbol)
        self.p2.setSymbol(self.p2_symbol)
        self.current_board = Board()
    
    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2
        
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_board = Board()
        self.p1.set_state(current_board)
        self.p2.set_state(current_board)

        if print_state:
            current_board.print()
        
        while True:
            player = next(alternator)
            move, symbol = player.act()
            next_board = current_board.act(move, symbol)
            next_board_hash = next_board.hash()
            current_board, isend = allstate[next_board_hash]
            self.p1.set_state(current_board)
            self.p2.set_state(current_board)
            if print_state:
                current_board.print()
            if isend:
                result = current_board.isEnd()
                if result == Result.X_Wins:
                    return Cell.X
                elif result == Result.O_Wins:
                    return Cell.O
                else:
                    return 0  # Draw
            
class Player:
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.symbol = None
    
    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, board):
        self.states.append(board)
        self.greedy.append(True)

    def setSymbol(self, symbol):
        self.symbol = symbol
        for hash_val in allstate:
            board, isend = allstate[hash_val]
            if isend:
                result = board.isEnd()
                if (result == Result.X_Wins and self.symbol == Cell.X) or \
                   (result == Result.O_Wins and self.symbol == Cell.O):
                    self.estimations[hash_val] = 1.0
                elif result == Result.Draw:
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    def step(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i + 1]] - self.estimations[state]
            ) # công thức cập nhật cho Q-table: V{t} = V{t} + alpha*(V{t+1} - V{t})
            self.estimations[state] += self.step_size * td_error
    
    def act(self):
        board = self.states[-1]
        next_boards = []
        next_moves = []
        for move in board.get_valid_moves():
            next_moves.append(move)
            next_boards.append(board.act(move, self.symbol).hash())
        
        if np.random.rand() < self.epsilon:
            move = next_moves[np.random.randint(len(next_moves))]
            self.greedy[-1] = False
            return move, self.symbol
        
        values = []
        for hash_val, move in zip(next_boards, next_moves):
            values.append((self.estimations.get(hash_val, 0.0), move))
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        move = values[0][1]

        return move, self.symbol
    
    def save_policy(self):
        symbol_name = 'first' if self.symbol == Cell.X else 'second'
        os.makedirs('policy', exist_ok=True)
        with open(f'policy/policy_{symbol_name}.bin', 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        symbol_name = 'first' if self.symbol == Cell.X else 'second'
        with open(f'policy/policy_{symbol_name}.bin', 'rb') as f:
            self.estimations = pickle.load(f)

class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.board = None

    def reset(self):
        pass

    def set_state(self, board):
        self.board = board

    def setSymbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.board.print()
        while True:
            try:
                coord_input = input(f"Input your position (row col) [0-{config.BOARD_ROWS-1}] [0-{config.BOARD_COLS-1}]: ")
                row, col = map(int, coord_input.split())
                if 0 <= row < config.BOARD_ROWS and 0 <= col < config.BOARD_COLS:
                    move = row * config.BOARD_COLS + col
                    if self.board.isValid(move):
                        return move, self.symbol
                    else:
                        print("Position already taken. Try again.")
                else:
                    print(f"Invalid coordinates. Use values between 0 and {config.BOARD_ROWS-1}")
            except (ValueError, IndexError):
                print("Invalid input. Use format: row col")

def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == Cell.X:
            player1_win += 1
        if winner == Cell.O:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.step()
        player2.step()
        judger.reset()
    player1.save_policy()
    player2.save_policy()

def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == Cell.X:
            player1_win += 1
        if winner == Cell.O:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play(print_state=True)
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")
        
        play_again = input("Play again? (y/n): ").lower()
        if play_again not in ['y', 'yes', '']:
            break

if __name__ == '__main__':
    # train(int(1e5))
    # compete(int(1e3))
    play()