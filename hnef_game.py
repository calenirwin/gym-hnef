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

