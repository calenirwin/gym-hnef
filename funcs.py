# Written By: davidADSP 
# Adapted for Hnefatafl By: Tindur Sigurdason & Calen Irwin

# Reference:
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/blob/master/funcs.py

import numpy as np
from gym_hnef.envs import hnef_env
from gym_hnef import hnef_game, hnef_vars
from model import Residual_CNN
from agent import Agent
import config
import memory
import action_ids
import gym

# Method for playing a number of matches between two agents
# In: p1, p2 agents playing against each other, mem Memory object, 
#       episodes number of games to be played, turn_until_tau0 turns until the agents stop exploring
#       rule set set is historical because copenhagen isn't implemented, render mode for the rendering of the game
def play_matches(p1, p2, mem=None, episodes=config.EPISODES, turn_until_tau0=config.TURNS_UNTIL_TAU0, rule_set='historical', render_mode='terminal'):
    # create gym environment
    env = gym.make('gym_hnef:hnef-v0', rule_set=rule_set, render_mode=render_mode)
    #  libary of how many games are won by each player, as well as draws
    scores = {p1.name:0, 'draw':0, p2.name:0}
    # library containing the agents
    players = { 0 :{"agent": p1, "name":p1.name},
                1 : {"agent": p2, "name":p2.name}}

    # play the games
    for e in range(episodes):
        print("Starting Episode ",e+1,"...")
        state = env.reset()

        done = 0
        t = 0
        p1.mcts = None
        p2.mcts = None

        while done == 0:
            # number of turns taken
            t += 1

            # pick action
            if t < turn_until_tau0:
                action, pi = players[hnef_game.turn(state)]['agent'].act(state, 1)
            else:
                action, pi = players[hnef_game.turn(state)]['agent'].act(state, 0)

            # commit to the short term memory
            if mem != None:
                mem.commit_stmemory(state, pi)

            # take the action selected
            state, reward, done, info = env.step(action)

            # current player
            turn = info['turn']
            # other player
            other_turn = np.abs(int(turn) - 1)
            
            # if game is finished
            if done == 1:
                # print the final game state
                print('Episode Completed. Final Game State:\n', hnef_game.str(state))
                # commit to memory
                if mem != None:
                    for move in mem.stmemory:
                        if move['player_turn'] == other_turn:
                            move['value'] = reward
                        else:
                            move['value'] = 0

                    mem.commit_ltmemory()

                # add the score to the player who won
                if reward == 2:
                    scores['draw'] += 1
                elif turn == 1:
                    scores[p1.name] += 1
                    print('Attacker win')
                elif turn == 0:
                    scores[p2.name] += 1
                    print('Defender win')
    return scores, mem

def evaluate_agents(p1, p2, num_games=100, rule_set='historical', render_mode='terminal'):
    # create gym environment
    env = gym.make('gym_hnef:hnef-v0', rule_set=rule_set, render_mode=render_mode)
    #  libary of how many games are won by each player, as well as draws
    scores = {p1.name:0, 'draw':0, p2.name:0}
    # library containing the agents
    players = { 0 :{"agent": p1, "name":p1.name},
                1 : {"agent": p2, "name":p2.name}}

    all_end_states = []

    done = 0
    p1.mcts = None
    p2.mcts = None

    for game in range(num_games):
        state = env.reset()

        while done == 0:
            action, _ = players[hnef_game.turn(state)]['agent'].act(state, 0)

            state, reward, done, info = env.step(action)
            # current player
            turn = info['turn']
            
            if done == 1:
                # print the final game state
                print('Game ' + str(game) + ' Final State:\n' + hnef_game.str(state))
                # add the score to the player who won
                if reward == 2:
                    scores['draw'] += 1
                elif turn == 1:
                    scores[p1.name] += 1
                elif turn == 0:
                    scores[p2.name] += 1

                all_end_states.append(state)

    print("\nScores after " + str(num_games) + " games completed") 
    print(scores)

    return scores, all_end_states

