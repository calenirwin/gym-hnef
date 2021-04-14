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

        self.mcts.backpropagation(leaf, value, path)

    def act(self, state, tau):
        if self.mcts == None or monte.Node(state).id not in self.mcts.tree:
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

        possible_actions_ids = []

        for i in range(len(possible_actions)):
            possible_actions_ids.append(action_ids.get_id(possible_actions[i]))
        possible_actions_ids = np.array(possible_actions_ids)

        # not sure what is going on here
        mask = np.ones(logits.shape, dtype=bool)
        mask[possible_actions_ids] = False
        logits[mask] = -100

        # apply softmax
        odds = np.exp(logits)
        probabilities = odds / np.sum(odds)

        return ((values, probabilities, possible_actions, possible_actions_ids))

    def evaluate_leaf(self, leaf, value, done, path):
        if done == 0:
            value, probabilities, possible_actions, possible_actions_ids = self.get_predictions(leaf.state)
            probabilities = probabilities[possible_actions_ids] # what is going on here?
            
            for i, action in enumerate(possible_actions):
                # print(action)
                new_state, _, _ = hnef_game.simulate_step(leaf.state, action)

                if monte.Node(new_state).id not in self.mcts.tree:
                    node = monte.Node(new_state)
                else:
                    node = self.mcts.tree[new_state].id

                new_edge = monte.Edge(leaf, node, probabilities[i], action)
                leaf.edges.append((action, new_edge))

        return ((value, path))


    def get_action_values(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pi[action] = edge.metrics['N']**(1/tau)
            values[action] = edge.metrics['Q']
        
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
        self.mcts.root = self.mcts.tree[new_root].id

    def replay(self, ltmemory):
        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convert_to_input(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
								, 'policy_head': np.array([row['AV'] for row in minibatch])}
                                
            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)

			# self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			# self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1],4)) 
			# self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1],4)) 

            # plt.plot(self.train_overall_loss, 'k')
            # plt.plot(self.train_value_loss, 'k:')
            # plt.plot(self.train_policy_loss, 'k--')

            # plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

            # display.clear_output(wait=True)
            # display.display(pl.gcf())
            # pl.gcf().clear()
            # time.sleep(1.0)