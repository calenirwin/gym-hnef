# Written By: Tindur Sigurdason & Calen Irwin
# Written For: CISC-856 W21 (Reinforcement Learning) at Queen's U
# Last Modified Date: 2021-03-06 [CI]
# Purpose: 

# References:
# https://github.com/slowen/hnefatafl/blob/master/hnefatafl.py
# https://github.com/aigagror/GymGo

import numpy as np
from scipy import ndimage
from sklearn import preprocessing

import hnef_vars

def init_state(rule_set):
    """
    at the moment, 2 represents the king in the defender board,
    may want to change this later
    """
    if rule_set == "copenhagen":
        state = np.zeros((hnef_vars.NUM_CHNLS, 11, 11))
        attacker_layout = np.array([[0,0,0,1,1,1,1,1,0,0,0],
                                    [0,0,0,0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0],
                                    [1,0,0,0,0,0,0,0,0,0,1],
                                    [1,0,0,0,0,0,0,0,0,0,1],
                                    [1,1,0,0,0,0,0,0,0,1,1],
                                    [1,0,0,0,0,0,0,0,0,0,1],
                                    [1,0,0,0,0,0,0,0,0,0,1],
                                    [0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,1,0,0,0,0,0],
                                    [0,0,0,1,1,1,1,1,0,0,0]])
                            
        defender_layout = np.array([[0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,1,0,0,0,0,0],
                                    [0,0,0,0,1,1,1,0,0,0,0],
                                    [0,0,0,1,1,2,1,1,0,0,0],
                                    [0,0,0,0,1,1,1,0,0,0,0],
                                    [0,0,0,0,0,1,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0,0,0]])

        state[hnef_vars.ATTACKER] = attacker_layout
        state[hnef_vars.DEFENDER] = defender_layout
        return state
        
    elif rule_set == "historical":
        state = np.zeros((hnef_vars.NUM_CHNLS, 9, 9))
        attacker_layout = np.array([[0,0,0,1,1,1,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0],
                                    [1,0,0,0,0,0,0,0,1],
                                    [1,0,0,0,0,0,0,0,1],
                                    [1,0,0,0,0,0,0,0,1],
                                    [0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,1,1,1,0,0,0]])

        defender_layout = np.array([[0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,1,1,2,1,1,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0,0,0]])

        state[hnef_vars.ATTACKER] = attacker_layout
        state[hnef_vars.DEFENDER] = defender_layout
        return state
    else:
        print("*Error: Given rule set has not been implemented.\n Existing rule sets are:\n-copenhagen\n-historial")
        return -1

# In: state (current state), action (action taken by current player)
# Out: state (new state)
# At the end of this method we want to check whether the new state ends the game
def next_state(state, action):

    # assert that the action is valid

    # switch turns


# In: state (current state)
# Out: valid_moves (list of all valid moves)
#
# We want to see which pieces are able to move by seeing which values > 0 are next to zeros
# then we're going to collect all the indexes of the possible moves and create a list of tuples
# so that each one is ((pos_x, pox_y), (new_x, new_y))
def compute_valid_moves(state):
    board_size = state.shape[1]

    full_board = state[0] + state[1]

    corners = np.array([0,0], [0, board_size-1], [board_size-1, 0], [board_size-1, board_size-1])
    throne = np.array([board_size // 2, board_size // 2])

    turn = int(np.max(state[hnef_vars.TURN_CHNL]))

    valid_moves = []

    # checking whose turn it is, if 1 then white, otherwise it's blacks' move
    if turn:
        # white to move

        # now we want to check for every zero in a straight line from values > 0
        # 
        #  
        for i in range(board_size):
            for j in range(board_size):
                
                # pass if (i, j) is a corner or no piece belonging to the current player is there
                if (np.array([i, j]) in corners) or (not state[turn][i][j]):
                    pass               

                pos_x, pos_y = i, j

                # up, down, left, right
                directions = np.array([0, 0, 0, 0])

                # check the neighbours
                #      
                if not i == 0:
                    # if we can move up
                    directions[0] = 1
                    while pos_x > 0 and not state[turn][pos_x-1][j]:
                        
                        valid_moves.append(((i, j), ()))

                
                if not i == board_size-1:
                    # if we can move down
                    directions[1] = 1

                
                if not j == 0:
                    # if we can move left
                    directions[2] = 1

                
                if not j == board_size-1:
                    # if we can move right
                    directions[3] = 1





    else:
        # black to move






def action_size(state=None, board_size: int = None):
    """
    Figure out a way to get the total number of actions for a given rule set
    This works for Go but it is likely different for Hnef
    """
     if state is not None:
        m, n = state.shape[1:]
    elif board_size is not None:
        m, n = board_size, board_size
    else:
        raise RuntimeError('No argument passed')
    return m * n + 1

