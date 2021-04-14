import numpy as np
from gym_hnef.envs import hnef_env
from gym_hnef import hnef_game, hnef_vars
from model import Residual_CNN
from agent import Agent
import config
import memory
import action_ids
import gym


def play_matches(p1, p2, mem=None, episodes=config.EPISODES, turn_until_tau0=config.TURNS_UNTIL_TAU0, rule_set='historical', render_mode='terminal'):
    for e in range(episodes):
        # create gym environment
        env = gym.make('gym_hnef:hnef-v0', rule_set=rule_set, render_mode=render_mode)
        state = env.state
        done = 0
        t = 0
        p1.mcts = None
        p2.mcts = None
        scores = {p1.name:0, 'draw':0, p2.name:0}

        players = {1:{"agent": p1, "name":p1.name}
                        , 0: {"agent": p2, "name":p2.name}
                        }

        while done == 0:
            # Number of turns taken
            t += 1
            print(hnef_game.turn(state))
            # Pick action
            if t < turn_until_tau0:
                action, pi, MCTS_val, NN_val = players[hnef_game.turn(state)]['agent'].act(state, 1)
            else:
                action, pi, MCTS_val, NN_val = players[hnef_game.turn(state)]['agent'].act(state, 0)

            if mem != None:
                mem.commit_stmemory(state, pi)

            state, reward, done, info = env.step(action)
            print(t)
            print(state[0]+state[1])
            turn = info['turn']
            other_turn = np.abs(int(turn) - 1)
            state[hnef_vars.TURN_CHNL][0][0] = np.abs(turn - 1)
            if done == 1:
                if mem != None:
                    for move in mem.stmemory:
                        if move['player_turn'] == other_turn:
                            move['value'] = reward
                        else:
                            move['value'] = 0

                    mem.commit_ltmemory()


                if reward == 2:
                    scores['draw'] += 1
                elif turn == 1:
                    scores[p1.name] += 1
                elif turn == 0:
                    scores[p2.name] += 1
    return scores, mem