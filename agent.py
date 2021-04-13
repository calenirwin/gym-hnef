# References:
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/agent.py

import numpy as np
from matplotlib import pyplot as plt

from gym_hnef import hnef_game, hnef_vars
from gym_hnef.envs import hnef_env

import config
import mcts as monte

# class User():
# 	def __init__(self, name, state_size, action_size):
# 		self.name = name
# 		self.state_size = state_size
# 		self.action_size = action_size

# 	def act(self, state, tau):
# 		action = input('Enter your chosen action: ')
# 		pi = np.zeros(self.action_size)
# 		pi[action] = 1
# 		value = None
# 		NN_value = None
# 		return (action, pi, value, NN_value)


class Agent():
    def __init__(self, name, model, state_size, action_size):
        super().__init__()

        self.name = name
        self.model = model
        self.mcts = None

        self.state_size = state_size
        self.action_size = action_size
        self.num_sims = config.MCTS_SIMS
        self.cpuct = config.CPUCT

        # self.train_loss = []
        # self.train_value_loss = []
        # self.train_policy_loss = []
        # self.val_loss = []
        # self.val_value_loss = []
        # self.val_policy_loss = []

    def simulate(self):
        # move to terminal node and evaluate
        leaf, value, done, path = self.mcts.traverse_tree()
        value, path = self.evaluate_leaf(leaf, value, done, path)

        self.mcts.backpropagation(leaf, value)

    def act(self, state, tau):
        if self.mcts == None or state.id not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.change_root_mcts(state)

        for sim in range(self.num_sims):
            self.simulate()

        pi, values, = self.get_action_values(tau=1)

        action, value = self.choose_action(pi, values, tau)

        next_state, _, _ = hnef_game.simulate_step(state, action)

        NN_value = -self.get_predictions(next_state)[0]

        return (action, pi, value, NN_value)

    def get_predictions(self, state):
        model_input = np.array(self.model.convert_to_input(state))

        predictions = self.model.predict(model_input)

        all_values = predictions[0]
        all_logits = predictions[1]

        values = all_values[0]
        logits = all_logits[0]

        possible_actions = hnef_game.compute_valid_moves(state)

        # not sure what is going on here
        mask = np.ones(logits.shape, dtype=bool)
        mask[possible_actions] = False
        logits[mask] = -100

        # apply softmax
        odds = np.exp(logits)
        probabilities = odds / np.sum(odds)

        return ((value, probabilities, possible_actions))

    def evaluate_leaf(self, leaf, value, done, path):
        if done == 0:
            value, probabilities, possible_actions = self.get_predictions(leaf.state)

            probabilities = probabilities[possible_actions] # what is going on here?

            for i, action in enumerate(possible_actions):
                new_state, _, _ = hnef_game.simulate_step(leaf.state, action)

                if new_State.id not in self.mcts.tree:
                    node = monte.Node(new_state)
                else:
                    node = self.mcts.tree[new_state.id]

                new_edge = monte.Edge(leaf, node, probabilities[i], action)
                leaf.edges.append((action, new_edge))

        return ((value, path))


    def get_action_values(self, tau):
        actions = hnef_game.compute_valid_moves(state)
        edges = self.mcts.edges
        pi = np.zeros(len(actions), dtype=np.integer)
        values = np.zeros(len(actions), dtype=np.float32)

        for action, edge in edges:
            pi[action] = edge.stats['N']**(1/tau)
            values[action] = edge.stats['Q']
        
        pi /= (np.sum(pi) * 1.0)

        return pi, values

    def choose_action(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = np.random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0]
        
        value = values[action]
        
        return action, value
            
    def predict(self, model_input):
        return self.model.predict(model_input)

    def build_mcts(self, state):
        self.root = monte.Node(state)
        self.mcts = monte.MCTS(self.root, self.cpuct)

    def change_root_mcts(self, state):
        new_root = monte.Node(state)
        self.mcts.root = self.mcts.tree[new_root.id]