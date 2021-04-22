# Purpose: Implement a Monte Carlo Search Tree algorithm for the purposes of training an 
# agent to learn how to play Hnefatafl aka Viking Chess. This code was modified from the given
# reference in an attempt to recreate the same approach used to implement famous AlphaZero architecture.

# Written By: davidADSP 
# Adapted for Hnefatafl By: Tindur Sigurdason & Calen Irwin

# References: 
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/MCTS.py

import numpy as np
import random
import string

from gym_hnef import hnef_game, hnef_vars
from gym_hnef.envs import hnef_env
import config

# Class to represent game states (as Nodes) in the Monte Carlo Search Tree
class Node():
    def __init__(self, state):
        self.state = state
        self.id = self.get_state_id(state)
        self.turn = hnef_game.turn(state)

        self.edges = []

    def __str__(self):
        return "Node ID: " + self.id + "\nPlayer's Turn: " + str(self.turn) + "\nNumber of Edges: " + str(len(self.edges))
        
    def is_leaf(self):
        # returns true if edges list is empty
        if len(self.edges) == 0:
            return True
        else:
            return False

    # Method that sets the id for each node equal to the board positions of the given state
    # associated with a node
    def get_state_id(self, state):
        position = state[hnef_vars.ATTACKER] + state[hnef_vars.DEFENDER]
        id = ''.join(map(str,position)) + str(state[hnef_vars.TIME_CHNL, 0, 0])
        return id

    def set_node_id(self, id):
        self.id = id

# Class to represent connections between game states (Edges)
# Each edge is directed meaning that there is a source node and
# a destination node, i.e., state 1 (source) + action -> state 2 (dest)
class Edge():
    def __init__(self, source, dest, prior, action):
        self.source = source    # input node
        self.dest = dest        # output node
        self.turn = hnef_game.turn(source.state)    # current players turn
        self.action = action    # action taken to get from source to dest
        
        # N: How many times has this action been taken?
        # W: Total value for the next state
        # Q: Mean value for the next state
        # P: Probability of picking this action
        self.metrics = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }

# Class that represents a Monte Carlo Search tree with 
# a dict representing the tree itself, containing Nodes and Edges
class MCTS():
    def __init__(self, root):
        self.root = root
        self.tree = {}
        # cpuct is a constant that helps control the balance between exploring
        # the tree and exploiting the discovered paths
        # Cp in UCT (Upper confidence bound for tree)
        self.cpuct = config.CPUCT
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def __str__(self):
        return "Root: " + str(self.root) + "\nTree Length: "  + str(len(self))  

    # Method to traverse/build the tree by simulating actions with the highest expected value
    # and keeping track of the actions taken
    def traverse_tree(self):
        alpha = config.ALPHA    # learning rate

        done = 0    # environment termination criteria
        value = 0   # holds predicted value of next state
        path = []   # holds edges taken during tree traversal
        prev_actions = []   # used to detect repetition problem

        current_node = self.root    # always begin traversal at the root of the tree

        # until you reach the end of the tree (no more actions can be taken)
        while not current_node.is_leaf():
            if current_node == self.root:
                epsilon = config.EPSILON
                NU = np.random.dirichlet([alpha] * len(current_node.edges)) # dirichlet distributed random variables
            else:
                epsilon = 0
                NU = [0] * len(current_node.edges)

            NB = 0
            for action, edge in current_node.edges:
                NB = NB + edge.metrics['N']

            max_QU = float('-inf')
            # loop through all actions to find the action that will maximize the expected value
            for i, (action, edge) in enumerate(current_node.edges):
                # calculate upper bound of for state value approximation
                U = self.cpuct * ((1 - epsilon) * edge.metrics['P'] + epsilon * NU[i]) * np.sqrt(NB) / (1 + edge.metrics['N'])
                Q = edge.metrics['Q']
                # set the next simulated action/edge pair as the action/edge that produces the highest value for the resulting state
                if Q + U > max_QU:
                    max_QU = Q + U
                    next_simulated_action = action
                    next_simulated_edge = edge

            prev_actions.append(action) # keep track of all simulated actions chosen

            # check to see if that last 6 actions were repetitions
            if len(prev_actions) > 6:
                this_last   = prev_actions[-1]
                this_next   = prev_actions[-3]
                this_first  = prev_actions[-5]
                other_last  = prev_actions[-2]
                other_next  = prev_actions[-4]
                other_first = prev_actions[-6]
                # if a repetition is found, finish the tree traversal
                if (np.mean(this_last == this_next) == 1 and np.mean(this_last == this_first) == 1) and (np.mean(other_last == other_next) == 1 and np.mean(other_last == other_first) == 1):
                    new_state, value, done = hnef_game.simulate_step(current_node.state, next_simulated_action)
                    current_node = next_simulated_edge.dest
                    path.append(next_simulated_edge)
                    print("***Repitition condition met in MCTS")
                    return current_node, value, done, path


            new_state, value, done = hnef_game.simulate_step(current_node.state, next_simulated_action)    
            current_node = next_simulated_edge.dest # new current node is the destination of the next simulated action
            path.append(next_simulated_edge)    # store the edge taken

        return current_node, value, done, path

    # Method that update the edges contained within path with
    # the results of the previous tree traversal
    # In: node that MCTS terminated at, value of the outcome of the tree traversal, path taken during tree traversal
    # Out: None
    def backpropagation(self, leaf_node, value, path):
        current_player = hnef_game.turn(leaf_node.state)

        for edge in path:
            turn = edge.turn

            if turn == current_player:
                direction = 1
            else:
                direction = -1

            edge.metrics['N'] += 1
            edge.metrics['W'] = edge.metrics['W'] + value * direction
            edge.metrics['Q'] = edge.metrics['W'] / edge.metrics['N']

    # Method to add a new node to the tree
    def add_node(self, node):
        self.tree[node.id] = node