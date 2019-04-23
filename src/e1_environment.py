# Tic Tac Toe board and game
import random


def set_up():
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if random.randint(0, 1) == 0:
        return board, "agent_1"
    else:
        return board, "agent_2"


def make_move(move, board, turn):
    win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    board_next = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(board)):
        board_next[i] = board[i]

    if board_next[move] == 0:
        board_next[move] = turn
    else:
        return False, -1, False

    for a, b, c in win_combinations:
        if board_next[a] == board_next[b] == board_next[c] == 1:
            return board_next, 1, False
        elif board_next[a] == board_next[b] == board_next[c] == -1:
            return board_next, -1, False
        elif board_next.count(0) == 0:
            return board_next, 0.5, False

    return board_next, 0, True


def change_board(board):
    if isinstance(board, list):
        for i in range(len(board)):
            board[i] = board[i] * -1
    return board
