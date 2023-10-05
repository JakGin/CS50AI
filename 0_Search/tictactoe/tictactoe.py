"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    parity = 0
    for row in board:
        for symbol in row:
            if symbol == "X":
                parity += 1
            elif symbol == "O":
                parity -= 1

    return "O" if parity else "X"


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == EMPTY:
                actions.add((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    error = False
    if action[0] < 0 or action[0] >= len(board):
        error = True
    elif action[1] < 0 or action[1] >= len(board[0]):
        error = True
    elif board[action[0]][action[1]] != EMPTY:
        error = True
    if error:
        raise Exception("Wrong action")

    result_board = deepcopy(board)
    result_board[action[0]][action[1]] = player(board)

    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winner = None

    # Check rows
    if board[0][0] == board[0][1] == board[0][2] != EMPTY:
        winner = board[0][0]
    elif board[1][0] == board[1][1] == board[1][2] != EMPTY:
        winner = board[1][0]
    elif board[2][0] == board[2][1] == board[2][2] != EMPTY:
        winner = board[2][0]

    # Check columns
    elif board[0][0] == board[1][0] == board[2][0] != EMPTY:
        winner = board[0][0]
    elif board[0][1] == board[1][1] == board[2][1] != EMPTY:
        winner = board[0][1]
    elif board[0][2] == board[1][2] == board[2][2] != EMPTY:
        winner = board[0][2]

    # Check diagonals
    elif board[0][0] == board[1][1] == board[2][2] != EMPTY:
        winner = board[0][0]
    elif board[0][2] == board[1][1] == board[2][0] != EMPTY:
        winner = board[0][2]

    return winner


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    for row in board:
        for symbol in row:
            if symbol == EMPTY:
                return False
    return True
    

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    result = winner(board)
    if result == "X":
        return 1
    if result == "O":
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    p = player(board)

    if p == "X":
        highest_value = float('-inf')
        for action in actions(board):
            value = min_value(result(board, action), float('-inf'), float('inf'))
            if value == 1:
                return action
            if value > highest_value:
                highest_value = value
                best_action = action
    elif p == "O":
        lowest_value = float('inf')
        for action in actions(board):
            value = max_value(result(board, action), float('-inf'), float('inf'))
            if value == -1:
                return action
            if value < lowest_value:
                lowest_value = value
                best_action = action
    
    return best_action


def max_value(state, alpha, beta):
    v = float("-inf")

    if terminal(state):
        return utility(state)
    
    for action in actions(state):
        v = max(v, min_value(result(state, action), alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)

    return v


def min_value(state, alpha, beta):
    v = float("inf")

    if terminal(state):
        return utility(state)
    
    for action in actions(state):
        v = min(v, max_value(result(state, action), alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)

    return v
