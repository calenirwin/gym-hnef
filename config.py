#### SELF PLAY
EPISODES = 1#0
MCTS_SIMS = 3#0
MEMORY_SIZE = 300#00
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
	{'filters':64, 'kernel_size': (3, 3)}
	 , {'filters':64, 'kernel_size': (3, 3)}
	#  , {'filters':64, 'kernel_size': (3, 3)}
	#  , {'filters':64, 'kernel_size': (3, 3)}
	#  , {'filters':64, 'kernel_size': (3, 3)}
	#  , {'filters':64, 'kernel_size': (3, 3)}
	]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 55/45