# Written By: davidADSP with minor changes from Tindur Sigurdarson and Calen Irwin
# For: CISC-856 W21 (Reinforcement Learning) at Queen's U
# Purpose: Contains a memory class, with both long and short term memory capabilities

# Reference: https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/memory.py

import numpy as np
from collections import deque

import config

class Memory:
	# Initialize the memory object
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = config.MEMORY_SIZE
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	# Commit to the short term memory
	def commit_stmemory(self, state, action_values):
		self.stmemory.append({
			'board': state[0] + state[1]
			, 'state': state
			, 'AV': action_values
			, 'player_turn': state[2, 0, 0]
			})

	# Commit to the long term memory, clears the short term memory after
	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	# Clears the short term memory
	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)