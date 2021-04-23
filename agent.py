# Written By: davidADSP 
# Adapted for Hnefatafl By: Tindur Sigurdason & Calen Irwin

# References:
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/agent.py

import numpy as np
import random
from matplotlib import pyplot as plt

from gym_hnef import hnef_game, hnef_vars
from gym_hnef.envs import hnef_env

import config
import mcts as monte
from mcts import Node

import action_ids
import small_action_space as small_action_ids


class RandomAgent():
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    # Method for the random agent to pick a valid action randomly
    def act(self, state, tau):
        valid_moves = hnef_game.compute_valid_moves(state)
        return valid_moves[random.randrange(len(valid_moves))], None

class Agent():
    def __init__(self, name, model, state_size, action_size):
        super().__init__()

        self.name = name
        self.model = model
        self.mcts = None

        self.state_size = state_size
        self.action_size = action_size
        self.num_sims = config.MCTS_SIMS

    # Method for running simulations from the current state
    #       moving to a terminal node, evaluating the leaf and updating the MCTS
    def simulate(self):
        leaf, value, done, path = self.mcts.traverse_tree()
        # print("Path length after tree traversal: " + str(len(path)))
        
        value, path = self.evaluate_leaf(leaf, value, done, path)
        # print("Path length after eval leaf: " + str(len(path)))
        
        self.mcts.backpropagation(leaf, value, path)
        # print("Path length after backprop: " + str(len(path)))

    # Method for running simulations and choosing an action
    # In: self, state (current state), tau (exploratory constant)
    # Out: action selected, current policy and values as well as values from the neural network
    def act(self, state, tau):
        # print("Initial State in Act: ", hnef_game.str(state))
        if self.mcts == None or monte.Node(state).id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            # print("I change the root!")
            self.change_root_mcts(state)
        
        for sim in range(self.num_sims):
            self.simulate()
            # self.mcts.print_tree()
        
        rootedges = []
        for e in self.mcts.root.edges:
            rootedges.append(e[1])

        pi, values, = self.get_action_values(tau=1)
        
        action, value = self.choose_action(pi, values, tau)
        
        if self.action_size == 625:
            action = small_action_ids.action_id[action]
        else:
            action = action_ids.action_id[action]

        self.mcts.all_states.append(state)

        next_state, _, _ = hnef_game.simulate_step(state, action)
        
        # NN_value = -self.get_predictions(next_state)[0]
        valid_moves = hnef_game.compute_valid_moves(state)

        if action not in valid_moves:
            for state in self.mcts.all_states:
                print(hnef_game.str(state))
            assert False
            
            self.fix_leaf(self.mcts.root)
            pi, values, = self.get_action_values(tau=1)
        
            action, value = self.choose_action(pi, values, tau)
            
            if self.action_size == 625:
                action = small_action_ids.action_id[action]
            else:
                action = action_ids.action_id[action]

        next_state, _, _ = hnef_game.simulate_step(state, action)

        return (action, pi)

    # Method for getting the predictions of values from the neural network
    # In: self, state (current state)
    # Out: values predicted, probabilities (policy), valid actions, id's of the valid actions
    def get_predictions(self, state):
        model_input = np.array(self.model.convert_to_input(state))

        predictions = self.model.predict(model_input)

        all_values = predictions[0]
        all_logits = predictions[1]

        values = all_values[0]
        logits = all_logits[0]

        possible_actions = hnef_game.compute_valid_moves(state)
        # print("State: \n" + hnef_game.str(state) + "Possible Actions:\n" + str(possible_actions))

        possible_actions_ids = []

        for i in range(len(possible_actions)):
            if self.action_size == 625:
                possible_actions_ids.append(small_action_ids.get_id(possible_actions[i]))
            else:
                possible_actions_ids.append(action_ids.get_id(possible_actions[i]))
        possible_actions_ids = np.array(possible_actions_ids)

        # print('Possible actions: ', len(possible_actions))

        # not sure what is going on here
        mask = np.ones(logits.shape, dtype=bool)
        for i in possible_actions_ids:
            mask[i] = False
        logits[mask] = -100

        # apply softmax
        odds = np.exp(logits)
        probabilities = odds / np.sum(odds)

        return ((values, probabilities, possible_actions, possible_actions_ids))
    
    def fix_leaf(self, leaf):
        print('I fix the leaf')
        state_copy = np.copy(leaf.state)
        leaf.edges = []
        values, probabilities, possible_actions, possible_actions_ids = self.get_predictions(state_copy)

        for i, action in enumerate(possible_actions):
            new_state, _, _ = hnef_game.simulate_step(np.copy(leaf.state), action)
            # if the node doesn't already exist in the tree, create it
            new_node_id = str(hash(str([new_state[0], new_state[1]])))
            
            if new_node_id not in self.mcts.tree:
                    node = monte.Node(new_state)
                    self.mcts.add_node(node)
            else:
                # print("I am a node that already exists: " + new_node_id)
                node = self.mcts.tree[new_node_id]

            # set the source node as the leaf and the dest node aka 'node' as the state of a given action
            new_edge = monte.Edge(leaf, node, probabilities[i], action)
            self.mcts.tree[leaf.id].edges.append((action, new_edge))

    # Method for evaluating a leaf, creates a new leaf node if the game isn't finished
    # In: leaf Node, value (reward), done boolean, path taken to the leaf
    # Out: value of the leaf node, path taken to the leaf node
    def evaluate_leaf(self, leaf, value, done, path):
        if done == 0:
            value, probabilities, possible_actions, possible_actions_ids = self.get_predictions(leaf.state)
            probs = []
            for i in possible_actions_ids:
                probs.append(probabilities[i])
            probabilities = probs 
            valid_moves = hnef_game.compute_valid_moves(leaf.state)
            if possible_actions != valid_moves:
                print('Turn:', hnef_game.turn(leaf.state))
                print('Possible actions: ', possible_actions)
                print('Actual valid moves: ', valid_moves )
            # print("Evaluating leaf:\n", leaf)
            # print("Value: ", value)
            # print("Probs: ",probabilities)
            # print("Actions: ",possible_actions)
            # assert False
            numedge = 0
            # loop through all possible actions at a given state
            # print('Possible actions: ', len(possible_actions))
            for i, action in enumerate(possible_actions):
                new_state, _, _ = hnef_game.simulate_step(np.copy(leaf.state), action)
                # if the node doesn't already exist in the tree, create it
                new_node_id = str(hash(str([new_state[0], new_state[1]])))

                if new_node_id not in self.mcts.tree:
                    node = monte.Node(new_state)
                    self.mcts.add_node(node)
                else:
                    # print("I am a node that already exists: " + new_node_id)
                    node = self.mcts.tree[new_node_id]

                # set the source node as the leaf and the dest node aka 'node' as the state of a given action
                new_edge = monte.Edge(leaf, node, probabilities[i], action)
                self.mcts.tree[leaf.id].edges.append((action, new_edge))
                numedge += 1
            # print('Number of edges added: ', numedge)
        return ((value, path))

    # Method for getting the action values of the possible actions in the curren state
    # In: self, tau (controls exploration)
    # Out: pi (policy), values of the actions
    def get_action_values(self, tau):
        edges = self.mcts.root.edges
        # print('Number of edges in the root:', len(edges))
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            if self.action_size == 625:
                action_id = small_action_ids.get_id(action)
            else:
                action_id = action_ids.get_id(action)
            pi[action_id] = np.power(edge.metrics['N'],(1/tau))
            values[action_id] = edge.metrics['Q']
        
        if np.sum(pi) != 0:
            pi = pi/float(np.sum(pi))

        return pi, values

    # Method for choosing an action from the given policy and values
    #       comparable to epsilon greedy but different
    # In: self, pi (policy), values of the actions, tau (controls exploration)
    # Out: action selected and its value
    def choose_action(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = np.random.choice(actions.reshape(-1))
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0]

        value = values[action]
        
        return action, value

    # Method for generating a prediction for a given state from the neural network
    def predict(self, model_input):
        return self.model.predict(model_input)

    # Method for generating our Monte Carlo Search Tree
    def build_mcts(self, state):
        self.root = monte.Node(state)
        self.mcts = monte.MCTS(self.root)

    # Method for changing the current root (state) in the MCTS
    def change_root_mcts(self, state):
        new_root = monte.Node(state)
        # print("Changed root:\n New root edges: " + str(len(self.mcts.root.edges)))
        self.mcts.root = self.mcts.tree[new_root.id]

    # Replays through the states in the long term memory and makes the neural network 
    #       learn from them
    def replay(self, ltmemory):
        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([row['state'] for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch]),
                                'policy_head': np.array([row['AV'] for row in minibatch])}
                                
            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)