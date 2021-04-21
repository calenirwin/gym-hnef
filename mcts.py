# References: 
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/MCTS.py

import numpy as np
import random
import string

from gym_hnef import hnef_game, hnef_vars
from gym_hnef.envs import hnef_env
import config

# Node class to represent game states in the MCTS
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

    # sets the id for each node equal to the board positions of the given state
    # associated with a node
    def get_state_id(self, state):
        position = state[hnef_vars.ATTACKER] + state[hnef_vars.DEFENDER]
        id = ''.join(map(str,position))#+str(state[hnef_vars.TIME_CHNL, 0, 0])
        return id

    def set_node_id(self, id):
        self.id = id

# Edge class to represent connections between game states
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

    def traverse_tree(self):
        alpha = config.ALPHA

        done = 0
        value = 0
        path = []

        prev_actions = []

        current_node = self.root
        count = 0
        inspect_flag = 0
        while not current_node.is_leaf():
            count += 1

            if count > 100:
                inspect_flag = 1

            if current_node == self.root:
                epsilon = config.EPSILON

                NU = np.random.dirichlet([alpha] * len(current_node.edges))
            else:
                epsilon = 0
                NU = [0] * len(current_node.edges)

            NB = 0
            for action, edge in current_node.edges:
                NB = NB + edge.metrics['N']

            max_QU = float('-inf')
            for i, (action, edge) in enumerate(current_node.edges):
                # print(epsilon)
                # print(edge.metrics)
                # print('NU:',len(NU))
                # print('i:',i)
                # print('current_node.edges length:',len(current_node.edges))
                # print(NB)
                U = self.cpuct * ((1 - epsilon) * edge.metrics['P'] + epsilon * NU[i]) * np.sqrt(NB) / (1 + edge.metrics['N'])

                Q = edge.metrics['Q']

                # print('Q:',Q, 'U:', U, 'Q+U:', Q+U, 'max_QU:', max_QU)
                # print("Q + U ", Q+U)
                # print("maxQU ", max_QU)
                if Q + U > max_QU:
                    max_QU = Q + U
                    next_simulated_action = action
                    next_simulated_edge = edge

            # print("state before action: ", current_node.state[0] + current_node.state[1])
            prev_actions.append(action)

            if len(prev_actions) > 6:
                this_last   = prev_actions[-1]
                this_next   = prev_actions[-3]
                this_first  = prev_actions[-5]
                other_last  = prev_actions[-2]
                other_next  = prev_actions[-4]
                other_first = prev_actions[-6]
                if (np.mean(this_last == this_next) == 1 and np.mean(this_last == this_first) == 1) and (np.mean(other_last == other_next) == 1 and np.mean(other_last == other_first) == 1):
                    new_state, value, done = hnef_game.simulate_step(current_node.state, next_simulated_action)
                    current_node = next_simulated_edge.dest
                    path.append(next_simulated_edge)
                    return current_node, value, done, path

            new_state, value, done = hnef_game.simulate_step(current_node.state, next_simulated_action)
            # print("state after action: ", new_state[0]+new_state[1])
            # print(next_simulated_edge.source.id == next_simulated_edge.dest.id)
            
            current_node = next_simulated_edge.dest
            path.append(next_simulated_edge)
            if inspect_flag:   
                return current_node, value, done, path
           # print(current_node)

        # print("I left this function!!!")
        return current_node, value, done, path

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

    def add_node(self, node):
        self.tree[node.id] = node