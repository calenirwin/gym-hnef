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
                                    [1,1,0,0,0,0,0,1,1],
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

def turn(state):
    if state is not None:
        return int(np.max(state[hnef_vars.TURN_CHNL]))

# Method for checking whether a capture has taken place
# In: state (current state), action (action taken by current player)
# Out: state (new state)
def check_capture(state, action):
    # current player
    current_player = turn(state)
    # other player
    other_player = np.abs(current_player - 1)
    # defender
    df = hnef_vars.DEFENDER
    # attacker
    at = hnef_vars.ATTACKER
    # board size
    board_size = state.shape[1]
    # throne location
    throne = (board_size // 2, board_size // 2)

    # new location of moved piece
    x, y = action[1]

    ## capturing normal pieces normally
    
    # capturing upwards
    if x > 1 and state[other_player][x-1][y] == 1 and state[current_player][x-2][y] > 0:
        state[other_player][x-1][y] = 0

    # capturing downwards
    if x < board_size - 2 and state[other_player][x+1][y] == 1 and state[current_player][x+2][y] > 0:
        state[other_player][x+1][y] = 0
    
    # capturing left
    if y > 1 and state[other_player][x][y-1] == 1 and state[current_player][x][y-2] > 0:
        state[other_player][x][y-1] = 0

    # capturing right
    if y < board_size - 2 and state[other_player][x][y+1] == 1 and state[current_player][x][y+2] > 0:
        state[other_player][x][y+1] = 0
    
    ## capturing normal pieces with the throne
    
    # if the king is on the throne then the white pieces cant be captured in this way
    if not (current_player == at and state[df][throne[0]][throne[0]] == 2):
        # capturing upwards
        if x > 1 and state[other_player][x-1][y] == 1 and np.mean(state[current_player][x-2][y] == throne):
            state[other_player][x-1][y] = 0
        # capturing downwards
        elif x < board_size - 2 and state[other_player][x+1][y] == 1 and np.mean(state[current_player][x+2][y] == throne):
            state[other_player][x+1][y] = 0
        # capturing left
        elif y > 1 and state[other_player][x][y-1] == 1 and np.mean(state[current_player][x][y-2] == throne):
            state[other_player][x][y-1] = 0
        # capturing right
        elif y < board_size - 2 and state[other_player][x][y+1] == 1 and np.mean(state[current_player][x][y+2] == throne):
            state[other_player][x][y+1] = 0
    
    ## capturing the king normally

    # capturing upwards
    if x > 1 and state[df][x-1][y] == 2 and state[current_player][x-2][y] > 0:
        state[df][x-1][y] = 0
        state[hnef_vars.DONE_CHNL] = 1
        return state

    # capturing downwards
    if x < board_size - 2 and state[df][x+1][y] == 2 and state[current_player][x+2][y] > 0:
        state[df][x+1][y] = 0
        state[hnef_vars.DONE_CHNL] = 1
        return state
    
    # capturing left
    if y > 1 and state[df][x][y-1] == 2 and state[current_player][x][y-2] > 0:
        state[df][x][y-1] = 0
        state[hnef_vars.DONE_CHNL] = 1
        return state

    # capturing right
    if y < board_size - 2 and state[df][x][y+1] == 2 and state[current_player][x][y+2] > 0:
        state[df][x][y+1] = 0
        state[hnef_vars.DONE_CHNL] = 1
        return state

    ## capturing the king on the throne
    if (state[df][throne[0]][throne[0]] == 2 
        and state[at][throne[0]-1][throne[0]] 
        and state[at][throne[0]+1][throne[0]] 
        and state[at][throne[0]][throne[0]-1] 
        and state[at][throne[0]][throne[0]]+1):
        state[df][throne[0]][throne[0]] = 0
        state[hnef_vars.DONE_CHNL] = 1
        return state

    ## capturing the king next to the throne

    if current_player == at:
        # king is above throne
        if state[df][throne[0]-1][throne[0]] == 2 and state[at][throne[0]-1][throne[0]-1] and state[at][throne[0]-1][throne[0]+1] and state[at][throne[0]-2][throne[0]]:
            state[df][throne[0]-1][throne[0]] = 0
            state[hnef_vars.DONE_CHNL] = 1
            return state
        # king is below throne  
        elif state[df][throne[0]+1][throne[0]] == 2 and state[at][throne[0]+1][throne[0]-1] and state[at][throne[0]+1][throne[0]+1] and state[at][throne[0]+2][throne[0]]:
            state[df][throne[0]+1][throne[0]] = 0
            state[hnef_vars.DONE_CHNL] = 1
            return state
        # king is left of throne 
        elif state[df][throne[0]][throne[0]-1] == 2 and state[at][throne[0]-1][throne[0]-1] and state[at][throne[0]+1][throne[0]-1] and state[at][throne[0]][throne[0]-2]:
            state[df][throne[0]][throne[0]-1] = 0
            state[hnef_vars.DONE_CHNL] = 1
            return state
        # king is right of throne 
        elif state[df][throne[0]][throne[0]+1] == 2 and state[at][throne[0]-1][throne[0]+1] and state[at][throne[0]+1][throne[0]+1] and state[at][throne[0]][throne[0]+2]:
            state[df][throne[0]][throne[0]+1] = 0
            state[hnef_vars.DONE_CHNL] = 1
            return state

    return state

# In: state (current state), action (action taken by current player)
# Out: state (new state)
# At the end of this method we want to check whether the new state ends the game
def next_state(state, action):

    # How will we handle repetitions?
    # we can either include the last two board positions in the state variable or keep track of the "timestamp"
    # I think the timestamp is a good way to handle this because with the same variable we can stop the game if
    # it gets too long, the only question is whether is should be a part of the state or a part of this class

    # define the current player
    current_player = turn(state)
    
    # assert that the action is valid i.e. that the action is in state[valid_actions]
    valid_moves = compute_valid_moves(state)
    print(valid_moves)
    print(action in valid_moves)
    print(action)
    assert action in valid_moves

    if state[current_player][action[0][0]][action[0][1]] == 2:
        state[current_player][action[0][0]][action[0][1]] = 0
        state[current_player][action[1][0]][action[1][1]] = 2
    else:
        state[current_player][action[0][0]][action[0][1]] = 0
        state[current_player][action[1][0]][action[1][1]] = 1

        

    # check if the player just captured a piece and update the state if so
    state = check_capture(state, action)

    # check if game is over

    # switch turns
    state[hnef_vars.TURN_CHNL][0][0] = np.abs(current_player - 1)

    # update state[valid_actions] for next player
    valid_moves = compute_valid_moves(state)

    return state

# In: state (current state), x-position of piece, y-position of piece
# Out: list of all possible actions where action a = ((x, y), (new_x, new_y))
def actions_for_piece(state, x, y):
    actions = []

    board_size = state.shape[1]

    # the position of every piece
    full_board = state[hnef_vars.ATTACKER] + state[hnef_vars.DEFENDER]

    throne = (board_size // 2, board_size // 2)
    # can the piece move up?
    if x > 0:
        pos_x = x
        pos_y = y 
        # continue until on the edge or about to collide with another piece
        while pos_x > 0 and not full_board[pos_x - 1, y]:
            pos_x -= 1
            # the action isn't possible if the destination is the throne, except if the piece is the king
            if ((full_board[x, y] == 2 and ((pos_x, y) == throne))) or (((pos_x, y) != throne)):
                actions.append(((x, y), (pos_x, y)))

    # can the piece move down?
    if x < board_size - 1:
        pos_x = x
        pos_y = y
        # continue until on the edge or about to collide with another piece
        while pos_x < board_size - 1 and not full_board[pos_x + 1, y]:
            pos_x += 1

            if ((full_board[x, y] == 2 and ((pos_x, y) == throne))) or (((pos_x, y) != throne)):
                actions.append(((x, y), (pos_x, y)))

    # can the piece move left?
    if y > 0:
        pos_x = x
        pos_y = y
        while pos_y > 0 and not full_board[x, pos_y - 1]:
            pos_y -= 1

            if ((full_board[x, y] == 2 and ((x, pos_y) == throne))) or (((x, pos_y) != throne)):
                actions.append(((x, y), (x, pos_y)))
                
    # can the piece move right?
    if y < board_size - 1:
        pos_x = x
        pos_y = y
        while pos_y < board_size - 1 and not full_board[x, pos_y + 1]:
            pos_y += 1

            if ((full_board[x, y] == 2 and ((x, pos_y) == throne))) or (((x, pos_y) != throne)):
                actions.append(((x, y), (x, pos_y)))

    return actions


# In: state (current state)
# Out: list of all possible actions for all pieces of the current player 
#      where action a = ((x, y), (new_x, new_y))
def compute_valid_moves(state):
    actions = []

    board_size = state.shape[1]

    current_player = turn(state)

    for i in range(board_size):
            for j in range(board_size):
                if state[current_player, i, j]:
                    piece_actions = actions_for_piece(state, i, j)

                    # this isn't the most efficient way of doing this 
                    # but I wanted to have a seperate helper method
                    for a in piece_actions:
                        actions.append(a)
    return actions

# In: state (current state), action (possible actions for current player)
# Out: list of all possible actions for all pieces of the current player 
#      where action a = ((x, y), (new_x, new_y))
# note: because actiosn is only the list of actions for the current player
# this check should only be done when it is the defenders turn
# unless we change that later, which we might want to do
def check_enclosure(state, action):
    board_size = state.shape[1] # get board size
    wall_positions = [] # holds wall positions for specific board size

    # populate wall positions array with tuples of (row_index, col_index)
    for i in range(board_size): # row index loop
        for j in range(board_size): # col index loop
            # bottom or top wall indices
            if i == 0 or i == board_size - 1:
                wall_positions.append((i,j))
            # side walls
            elif j == 0 or j == board_size -1:
                wall_positions.append((i,j))

    # list comprehension to check if a wall position is within the list of 
    # possible actions for the defender
    # appends those locations if they exist in the action list
    # wall_moves = [pos for pos in wall_positions if pos in action]

    # not as pretty but still works
    # should we break early or add some terminating condition?
    wall_moves = []
    
    for pos in wall_positions:
        for a in action:
            if a[1] == pos:
                wall_moves.append(pos)

    # if no defender pieces can move to a wall then they are either enclosed or 
    # there are no remaining defender pieces
    # either way the game is over and the attacker wins
    if len(wall_moves) == 0:
        return True, hnef_vars.ATTACKER
    else:
        return False, -1

def is_over(state, action):
    if (state is not None):
        at = hnef_vars.ATTACKER
        df = hnef_vars.DEFENDER
        full_board = state[at] + state[df]
        board_size = state.shape[1]    
        # current player
        player = int(np.max(state[hnef_vars.TURN_CHNL]))
        # other player
        other_player = np.abs(player - 1)

        # has the king been captured?
        if np.max(state[df]) < 2:
            print("***King captured")
            return True, at
        # has the king escaped?
        elif np.max(state[df][0]) == 2 or np.max(state[df][:,0]) == 2 or np.max(state[df][:,board_size-1]) == 2 or np.max(state[df][board_size-1]) == 2:
            print("***King escaped")
            return True, df
        # has the attacker enclosed the defender?
        # elif check_enclosure(state,action)[0]:
        #     return True, at
        # no win
        else:
            return False, -1
    else:
        return False, 1

def simulate_step(state, action):
    new_state = next_state(state, action)
    done, winner = is_over(state, action) 

    if not done:
            reward = 0
    else:
        current_player = turn(state)
        if current_player == winner:
            reward =  1
        else:
            reward = 0

    return np.copy(state), reward, done

def str(state):
    board_str = ' '

    size = state.shape[1]
    for i in range(size):
        board_str += '   {}'.format(i)
    board_str += '\n  '
    board_str += '----' * size + '-'
    board_str += '\n'
    for i in range(size):
        board_str += '{} |'.format(i)
        for j in range(size):
            if state[0, i, j] == 1:
                board_str += ' A'
            elif state[1, i, j] == 1:
                board_str += ' D'
            elif state[1, i, j] == 2:
                board_str += ' K'
            else:
                board_str += '  '

            board_str += ' |'

        board_str += '\n  '
        board_str += '----' * size + '-'
        board_str += '\n'

    t = turn(state)
    board_str += '\tTurn: {}'.format('Attacker' if t == 0 else 'Defender')
    return board_str