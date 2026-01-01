#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <limits>

// --- Constants & Configuration ---
// Updated to 20 based on your log output (indices went up to 19)
constexpr std::size_t BOARD_SIZE = 20;

// Score Constants
// We use a "Base" win score. The actual score will be BASE + DEPTH.
constexpr int EVAL_WIN_BASE = 1000000000; 
constexpr int EVAL_MIN = -EVAL_WIN_BASE - 100; // Lower than any possible loss
constexpr int EVAL_MAX = EVAL_WIN_BASE + 100;  // Higher than any possible win

// Heuristics (Must be significantly lower than EVAL_WIN_BASE)
constexpr int SCORE_OPEN_4   = 100000000; 
constexpr int SCORE_CLOSED_4 = 5000000;   
constexpr int SCORE_OPEN_3   = 1000000;   
constexpr int SCORE_CLOSED_3 = 10000;
constexpr int SCORE_OPEN_2   = 5000;
constexpr int SCORE_CLOSED_2 = 100;
constexpr int SCORE_OPEN_1   = 10;

enum CellState : std::int8_t {
    EMPTY = 0,
    X = 1,
    O = -1
};

// --- Board Class ---
class Board {
public:
    typedef std::size_t coor_t;

    Board() : cells_(), current_player_(X) {
        cells_.fill(EMPTY);
        // Start center-ish for 20x20
        play_ranges_.push_back({9, 11, 9, 11});
    }

    CellState get_cell(std::size_t row, std::size_t col) const {
        if (row >= BOARD_SIZE || col >= BOARD_SIZE) return O; 
        return cells_[index(row, col)];
    }

    CellState get_cell_unsafe(std::size_t row, std::size_t col) const {
        return cells_[index(row, col)];
    }

    void make_move(coor_t coor) {
        cells_[coor] = current_player_;
        play_ranges_.push_back(play_ranges_.back());
        play_ranges_.back().update(coor / BOARD_SIZE, coor % BOARD_SIZE);
        current_player_ = (current_player_ == X) ? O : X;
    }

    void make_move(std::size_t row, std::size_t col) {
        make_move(index(row, col));
    }

    void undo_move(coor_t coor) {
        current_player_ = (current_player_ == X) ? O : X;
        play_ranges_.pop_back();
        cells_[coor] = EMPTY;
    }

    void undo_move(std::size_t row, std::size_t col) {
        undo_move(index(row, col));
    }

    CellState current_player() const { return current_player_; }

    std::vector<coor_t> get_ordered_moves() const {
        std::vector<coor_t> moves;
        moves.reserve(200); // Heuristic reservation
        
        const PlayRange &range = play_ranges_.back();
        
        for (std::size_t r = range.min_r; r <= range.max_r; ++r) {
            for (std::size_t c = range.min_c; c <= range.max_c; ++c) {
                if (cells_[index(r, c)] == EMPTY) {
                    moves.push_back(index(r, c));
                }
            }
        }

        // Sort: Moves with neighbors first
        auto has_neighbor = [&](coor_t idx) {
            int r = idx / BOARD_SIZE;
            int c = idx % BOARD_SIZE;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < (int)BOARD_SIZE && nc >= 0 && nc < (int)BOARD_SIZE) {
                        if (cells_[index(nr, nc)] != EMPTY) return true;
                    }
                }
            }
            return false;
        };

        std::partition(moves.begin(), moves.end(), has_neighbor);
        return moves;
    }

    CellState check_winner() const {
        // Optimization: Only scan the play range + margin
        // But for safety/simplicity in this snippet, we scan relevant rows in play_range
        const PlayRange &range = play_ranges_.back();
        // Expand check range slightly to catch lines ending exactly at boundary
        size_t r_start = (range.min_r > 4) ? range.min_r - 4 : 0;
        size_t r_end   = (range.max_r < BOARD_SIZE - 5) ? range.max_r + 4 : BOARD_SIZE - 1;
        size_t c_start = (range.min_c > 4) ? range.min_c - 4 : 0;
        size_t c_end   = (range.max_c < BOARD_SIZE - 5) ? range.max_c + 4 : BOARD_SIZE - 1;

        const int dr[] = {0, 1, 1, 1};
        const int dc[] = {1, 0, 1, -1};

        for (std::size_t r = r_start; r <= r_end; ++r) {
            for (std::size_t c = c_start; c <= c_end; ++c) {
                CellState p = cells_[index(r,c)];
                if (p == EMPTY) continue;
                for (int i = 0; i < 4; ++i) {
                    int tr = r, tc = c, count = 0;
                    while(tr >= 0 && tr < (int)BOARD_SIZE && tc >= 0 && tc < (int)BOARD_SIZE && cells_[index(tr, tc)] == p) {
                        count++; tr += dr[i]; tc += dc[i];
                    }
                    if (count >= 5) return p;
                }
            }
        }
        return EMPTY;
    }

    friend std::ostream &operator<<(std::ostream &os, const Board &board) {
        os << "   ";
        for (std::size_t i = 0; i < BOARD_SIZE; ++i) {
            os << i;
            if (i < 10) os << " ";
            else os << ""; // Compact formatting for >10
        }
        os << "\n";
        for (std::size_t row = 0; row < BOARD_SIZE; ++row) {
            os << row << (row < 10 ? "  " : " ");
            for (std::size_t col = 0; col < BOARD_SIZE; ++col) {
                CellState cell = board.get_cell_unsafe(row, col);
                char c = (cell == X) ? 'X' : (cell == O) ? 'O' : '.';
                os << c << (col < 10 ? " " : "  "); // Adjust spacing
            }
            os << '\n';
        }
        return os;
    }

    std::vector<coor_t> get_empty_cells() const {
        std::vector<coor_t> empty_cells;
        for (std::size_t i = 0; i < cells_.size(); ++i)
            if (cells_[i] == EMPTY) empty_cells.push_back(i);
        return empty_cells;
    }

private:
    struct PlayRange {
        std::size_t min_r, max_r, min_c, max_c;
        void update(std::size_t row, std::size_t col) {
            min_r = (row > 1) ? std::min(min_r, row - 2) : 0;
            max_r = (row < BOARD_SIZE - 2) ? std::max(max_r, row + 2) : BOARD_SIZE - 1;
            min_c = (col > 1) ? std::min(min_c, col - 2) : 0;
            max_c = (col < BOARD_SIZE - 2) ? std::max(max_c, col + 2) : BOARD_SIZE - 1;
        }
    };
    std::array<CellState, BOARD_SIZE * BOARD_SIZE> cells_;
    CellState current_player_;
    std::vector<PlayRange> play_ranges_;
    constexpr std::size_t index(std::size_t row, std::size_t col) const { return row * BOARD_SIZE + col; }
};

// --- Evaluation ---
bool is_valid_pos(int r, int c) {
    return r >= 0 && r < (int)BOARD_SIZE && c >= 0 && c < (int)BOARD_SIZE;
}

int evaluate(const Board& board) {
    int score_x = 0;
    int score_o = 0;
    const int dr[] = {0, 1, 1, 1};
    const int dc[] = {1, 0, 1, -1};

    for (int r = 0; r < (int)BOARD_SIZE; ++r) {
        for (int c = 0; c < (int)BOARD_SIZE; ++c) {
            CellState type = board.get_cell_unsafe(r, c);
            if (type == EMPTY) continue;

            for (int dir = 0; dir < 4; ++dir) {
                int prev_r = r - dr[dir];
                int prev_c = c - dc[dir];
                if (is_valid_pos(prev_r, prev_c) && board.get_cell_unsafe(prev_r, prev_c) == type) continue;

                int count = 0;
                int curr_r = r, curr_c = c;
                while (is_valid_pos(curr_r, curr_c) && board.get_cell_unsafe(curr_r, curr_c) == type) {
                    count++; curr_r += dr[dir]; curr_c += dc[dir];
                }

                int open_ends = 0;
                if (is_valid_pos(prev_r, prev_c) && board.get_cell_unsafe(prev_r, prev_c) == EMPTY) open_ends++;
                if (is_valid_pos(curr_r, curr_c) && board.get_cell_unsafe(curr_r, curr_c) == EMPTY) open_ends++;

                int current_score = 0;
                if (count >= 5) current_score = EVAL_WIN_BASE; // Base win, handled by check_winner usually
                else if (count == 4) {
                    if (open_ends == 2) current_score = SCORE_OPEN_4;
                    else if (open_ends == 1) current_score = SCORE_CLOSED_4;
                } else if (count == 3) {
                    if (open_ends == 2) current_score = SCORE_OPEN_3;
                    else if (open_ends == 1) current_score = SCORE_CLOSED_3;
                } else if (count == 2) {
                    if (open_ends == 2) current_score = SCORE_OPEN_2;
                    else if (open_ends == 1) current_score = SCORE_CLOSED_2;
                } else if (count == 1 && open_ends == 2) {
                    current_score = SCORE_OPEN_1;
                }

                if (type == X) score_x += current_score;
                else score_o += current_score;
            }
        }
    }

    if (board.current_player() == X) return score_x - score_o;
    return score_o - score_x;
}

// --- Negamax with Depth-Scaled Scores ---
int negamax(Board& board, int depth, int alpha, int beta) {
    // 1. Terminal State Check: Winner
    CellState winner = board.check_winner();
    if (winner != EMPTY) {
        // If there is a winner, it's the player who Just Moved (not current_player).
        // So the current_player has LOST.
        // We return a negative score.
        // We subtract 'depth' so that a "fast loss" (high depth) is more negative than a "slow loss".
        // Conversely, the parent (winner) sees a less negative number negated -> larger positive number.
        return -(EVAL_WIN_BASE + depth);
    }
    
    // 2. Terminal State Check: Depth or Draw
    if (depth == 0) return evaluate(board);

    std::vector<Board::coor_t> moves = board.get_ordered_moves();
    if (moves.empty()) return 0;

    int max_eval = EVAL_MIN;
    for (auto move_idx : moves) {
        board.make_move(move_idx);
        int eval = -negamax(board, depth - 1, -beta, -alpha);
        board.undo_move(move_idx);

        if (eval > max_eval) max_eval = eval;
        if (eval > alpha) alpha = eval;
        if (alpha >= beta) break;
    }
    return max_eval;
}

std::size_t get_best_move(Board& board, int depth) {
    std::vector<Board::coor_t> moves = board.get_ordered_moves();
    if (moves.empty()) return (BOARD_SIZE * BOARD_SIZE) / 2;

    std::size_t best_move = moves[0];
    int max_eval = EVAL_MIN;
    int alpha = EVAL_MIN;
    int beta = EVAL_MAX;

    std::cout << "Thinking (Depth " << depth << ")... ";
    auto start = std::chrono::high_resolution_clock::now();

    for (auto move_idx : moves) {
        board.make_move(move_idx);
        int eval = -negamax(board, depth - 1, -beta, -alpha);
        board.undo_move(move_idx);

        // Found a forced win?
        if (eval >= EVAL_WIN_BASE) {
            std::cout << "Winning move found! (Score: " << eval << ")\n";
            best_move = move_idx;
            max_eval = eval;
            break; // Stop searching, take the win immediately
        }

        if (eval > max_eval) {
            max_eval = eval;
            best_move = move_idx;
        }
        if (eval > alpha) alpha = eval;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Done in " << elapsed.count() << "ms. (Eval: " << max_eval << ")\n";

    return best_move;
}

int main(int argc, char* argv[]) {
    int depth = 4;
    if (argc > 1) depth = std::stoi(argv[1]);

    Board board;
    std::cout << "Gomoku AI (Depth: " << depth << ", Board: " << BOARD_SIZE << "x" << BOARD_SIZE << ")\n";
    std::cout << "You are 'X'. Enter moves as 'row col'.\n";

    while (true) {
        std::cout << board;
        
        CellState winner = board.check_winner();
        if (winner != EMPTY) {
            std::cout << "Winner: " << (winner == X ? "Player (X)" : "AI (O)") << "\n";
            break;
        }
        if (board.get_empty_cells().empty()) break;

        if (board.current_player() == X) {
            int r, c;
            while (true) {
                std::cout << "Your move: ";
                if (std::cin >> r >> c && is_valid_pos(r, c) && board.get_cell(r, c) == EMPTY) break;
                std::cout << "Invalid.\n";
                std::cin.clear();
                std::cin.ignore(10000, '\n');
            }
            board.make_move(r, c);
        } else {
            std::size_t move = get_best_move(board, depth);
            std::cout << "AI plays: " << move / BOARD_SIZE << " " << move % BOARD_SIZE << "\n";
            board.make_move(move);
        }
    }
    return 0;
}