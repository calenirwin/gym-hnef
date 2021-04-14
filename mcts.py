# References: 
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/MCTS.py

import numpy as np

from gym_hnef import hnef_game, hnef_vars
from gym_hnef.envs import hnef_env

import config

class Node():
    def __init__(self, state):
        self.state = state
        self.id = self.get_state_id(state)
        self.turn = hnef_game.turn(state)

        self.edges = []
        
    def is_leaf(self):
        # returns true if edges list is empty
        if not self.edges:
            return True
        else:
            return False

    def get_state_id(self, state):
        # make each board state a unique id for each node
        position = np.append(state[hnef_vars.ATTACKER], state[hnef_vars.DEFENDER])
        id = ''.join(map(str,position))
        return id

    def set_node_id(self, id):
        self.id = id

class Edge():
    def __init__(self, source, dest, prior, action):
        self.source = source    # input node
        self.dest = dest        # output node
        self.turn = hnef_game.turn(source.state)
        self.action = action
        
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

class MCTS():
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        # cpuct is a constant that helps control the balance between exploring
        # the tree and exploiting the discovered paths
        # Cp in UCT (Upper confidence bound for tree)
        self.cpuct = config.CPUCT
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def traverse_tree(self):
        epsilon = config.EPSILON
        alpha = config.ALPHA

        done = False
        value = 0

        path = []

        current_node = self.root
        print('Is current node a leaf?', current_node.is_leaf())
        while not current_node.is_leaf():
            if current_node == self.root:
                epsilon = 0.2
                alpha = 0.8

                NU = np.random.dirichlet([alpha] * len(current_node.edges))
            else:
                epsilon = 0
                NU [0] * len(current_node.edges)

            NB = 0
            for action, edge in current_node.edges:
                NB = NB + edge.metrics['N']

            max_QU = -99999
            for i, (action, edge) in enumerate(current_node.edges):
                U = self.cpuct * ((1 - epsilon) * edge.metrics['P'] + epsilon * NU[i]) * np.sqrt(NB) / (1 + edge.metrics['N'])

                Q = edge.metrics['Q']

                if Q + U > max_QU:
                    max_QU = Q + U
                    next_simulated_action = action
                    next_simulated_edge = edge
                
            new_state, value, done = hnef_game.simulate_step(current_node.state, next_simulated_action)
            current_node = next_edge.dest
            path.append(current_node)

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
            # check to see if the node id already exists in tree
            if node.id in self.tree:
                new_id = node.id + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) # add 5 random characters to the end of the duplicate state id
                node.set_node_id(new_id)

            self.tree[node.id] = node