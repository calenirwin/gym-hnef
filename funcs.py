import numpy as np
from gym_hnef.envs import hnef_env
from gym_hnef import hnef_game, hnef_vars
from model import Residual_CNN
from agent import Agent, User
import config


def play_matches(p1, p2, episodes=config.EPISODES, turn_until_tau0=config.TURNS_UNTIL_TAU0, rule_set='historical'):
    for e in episodes:
        env = hnef_env.HnefEnv(rule_set)
        state = env.state
        done = 0
        turn = 0
        p1.mcts = None
        p2.mcts = None
        scores = {p1.name:0, 'draw':0, p2.name:0}

        players = {1:{"agent": p1, "name":p1.name}
                        , 0: {"agent": p2, "name":p2.name}
                        }

        while done == 0:
            turn += 1

            if turn < turn_until_tau0:
                action, pi, MCTS_val, NN_val = players[state[2, 0, 0]]['agent'].act(state, 1)
            else:
                action, pi, MCTS_val, NN_val = players[state[2, 0, 0]]['agent'].act(state, 0)
            
            state, r, done, turn = env.step(action)
            turn = turn['turn']
            other_turn = np.abs(int(turn) - 1)
            if done == 1:
                if turn == 1:
                    scores[p1.name] += 1
                elif turn == 2:
                    scores[p2.name] += 1
    return scores