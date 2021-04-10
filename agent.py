import numpy as np
from gym_hnef import hnef_game, hnef_vars
from gym_hnef.envs import hnef_env
from model import softmax_cross_entropy_with_logits
import config
from matplotlib import pyplot as plt

class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = input('Enter your chosen action: ')
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)


class Agent():
    def __init__(self, name, model):
        super().__init__()

        self.name = name
        self.model = model
        self.mcts = None

        self.train_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []
    
    # TODO
    def simulate(self):
        return False
    
    # TODO
    def act(self, state, tau):
        return False

    def get_action_values(self, state, tau):
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

    # TODO
    def build_mcts(self, state):
        return False

    # TODO
    def change_root_mcts(self, state):
        return False