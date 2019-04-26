# Tic Tac Toe board and game
import random


def set_up():
    board = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]
    # das brett, erste liste für agent 1, die zweite liste für agent 2, drite liste für die leeren felder
    # dh in jeder spalte muss eine 0.99 stehen und rest 0.01, damit nicht teil gegen null konvergiert
    if random.randint(0, 1) == 0:
        return board, "agent_1"
    else:
        return board, "agent_2"


def make_move(move, board, turn):
    win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    board_next = list(board)

    if board_next[2][move] == 0.99:
        board_next[turn][move] = 0.99
        board_next[2][move] = 0.01
    else:
        return False, -1, False

    for a, b, c in win_combinations:
        if board_next[turn][a] == board_next[turn][b] == board_next[turn][c] == 0.99:
            return board_next, 1, False
        elif board_next[abs(turn-1)][a] == board_next[abs(turn-1)][b] == board_next[abs(turn-1)][c] == 0.99:
            return board_next, -1, False
        elif board_next[2].count(1) == 0.01:
            return board_next, 0.5, False

    return board_next, 0, True
