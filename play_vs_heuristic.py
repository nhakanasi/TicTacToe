"""
Play Tic Tac Toe against a trained TD Learning agent.
Human plays as X (goes first), TD agent plays as O.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strat.encode import Board, Cell, Result, BOARD_ROWS, BOARD_COLS
from strat.heuristic import TDPlayer


class HumanPlayer:
    """Human player that reads moves from user input."""
    
    def __init__(self):
        self.symbol = None
        self.board = None
    
    def reset(self):
        """Reset player state for new game."""
        pass
    
    def set_board(self, board: Board):
        """Update player's board reference."""
        self.board = board
    
    def setSymbol(self, symbol: int):
        """Set player's symbol (Cell.X or Cell.O)."""
        self.symbol = symbol
    
    def act(self) -> int:
        """
        Get move from human player via user input.
        
        Returns:
            1D index of the move
            
        Raises:
            KeyboardInterrupt: If user enters 'quit', 'q', or 'exit'
        """
        self.board.print()
        
        while True:
            try:
                user_input = input("Enter your move (row col): ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'q', 'exit']:
                    raise KeyboardInterrupt
                
                # Parse input
                parts = user_input.split()
                if len(parts) != 2:
                    print("Invalid input! Please enter row and column separated by space (e.g., '0 1')")
                    continue
                
                row = int(parts[0])
                col = int(parts[1])
                
                # Validate coordinates
                if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
                    print(f"Invalid coordinates! Row and column must be between 0 and {BOARD_ROWS-1}")
                    continue
                
                # Convert to 1D index
                move_1d = row * BOARD_COLS + col
                
                # Check if position is empty
                if not self.board.isValid(move_1d):
                    print("Invalid move! Position already taken.")
                    continue
                
                return move_1d
                
            except ValueError:
                print("Invalid input! Please enter two numbers separated by space (e.g., '0 1')")
                continue
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise


class Judger:
    """
    Manages a game between a human player and TD agent.
    """
    
    def __init__(self, human_player: HumanPlayer, ai_player: TDPlayer):
        """
        Initialize the judger with two players.
        
        Args:
            human_player: HumanPlayer instance
            ai_player: TDPlayer instance (trained agent)
        """
        self.human = human_player
        self.ai = ai_player
        
        # Human plays X (goes first), AI plays O
        self.human_symbol = Cell.X
        self.ai_symbol = Cell.O
        
        self.human.setSymbol(self.human_symbol)
        self.ai.symbol = self.ai_symbol
    
    def reset(self):
        """Reset both players for new game."""
        self.human.reset()
    
    def alternate(self):
        """Alternate between human and AI player."""
        while True:
            yield self.human
            yield self.ai
    
    def play(self, print_state: bool = True) -> Result:
        """
        Play one complete game between human and AI.
        
        Args:
            print_state: Whether to print board after each move
            
        Returns:
            Game result (Result.X_Wins, Result.O_Wins, or Result.Draw)
        """
        alternator = self.alternate()
        self.reset()
        current_board = Board()
        
        self.human.set_board(current_board)
        self.ai.symbol = self.ai_symbol
        
        move_count = 0
        
        if print_state:
            print("\n" + "=" * 70)
            print("Tic Tac Toe: Human (X) vs TD Agent (O)")
            print("=" * 70)
            print("You are X (goes first). AI is O.")
            print("Enter moves as 'row col' (e.g., '0 0' for top-left)")
            print("=" * 70)
            current_board.print()
        
        while True:
            player = next(alternator)
            
            # Get move from current player
            try:
                if player == self.human:
                    move = self.human.act()
                    symbol_str = "X"
                else:
                    move = self.ai.get_move(current_board)
                    symbol_str = "O"
                    if move is not None:
                        row, col = divmod(move, BOARD_COLS)
                        print(f"\nAI plays at ({row}, {col})")
            except KeyboardInterrupt:
                return Result.Incomplete
            
            if move is None:
                break
            
            # Execute move
            current_board = current_board.act(move, player.symbol if player == self.human else self.ai_symbol)
            move_count += 1
            
            # Update human's board reference
            self.human.set_board(current_board)
            
            if print_state:
                current_board.print()
            
            # Check if game ended
            result = current_board.isEnd()
            if result != Result.Incomplete:
                break
        
        # Print result
        if print_state:
            print("=" * 70)
            if result == Result.X_Wins:
                print("🎉 Congratulations! You (X) win!")
            elif result == Result.O_Wins:
                print("😞 AI (O) wins! Better luck next time!")
            elif result == Result.Draw:
                print("🤝 It's a tie! Well played!")
            else:
                print("Game interrupted.")
            print("=" * 70 + "\n")
        
        return result


def main():
    """Main function to play human vs TD agent."""
    
    # Check if weights file exists
    weights_path = r"Reinforecedment learning\tictactoe\policy\td_agent_o.pkl"
    
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights not found at '{weights_path}'")
        print("Please train the agent first using:")
        print("  python -m strat.heuristic train")
        print("or")
        print("  cd strat && python heuristic.py")
        sys.exit(1)
    
    # Load trained AI agent
    print(f"Loading trained TD agent from '{weights_path}'...")
    ai_player = TDPlayer(weights_path, Cell.O)
    print("AI agent loaded successfully!\n")
    
    # Create human player
    human_player = HumanPlayer()
    
    # Create judger
    judger = Judger(human_player, ai_player)
    
    # Play games until user quits
    play_again = True
    game_count = 0
    human_wins = 0
    ai_wins = 0
    draws = 0
    
    while play_again:
        game_count += 1
        try:
            result = judger.play(print_state=True)
            
            # Update statistics
            if result == Result.X_Wins:
                human_wins += 1
            elif result == Result.O_Wins:
                ai_wins += 1
            elif result == Result.Draw:
                draws += 1
            
            # Ask to play again
            while True:
                try:
                    response = input("Play again? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        play_again = True
                        break
                    elif response in ['n', 'no']:
                        play_again = False
                        break
                    else:
                        print("Please enter 'y' or 'n'")
                except KeyboardInterrupt:
                    play_again = False
                    break
        
        except KeyboardInterrupt:
            play_again = False
    print("\nThanks for playing!")


if __name__ == "__main__":
    main()