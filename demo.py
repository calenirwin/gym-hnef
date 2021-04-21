# Purpose: Utilize Gym-Hnef environment to play Hnefatafl 
# using either the terminal or a rendered GUI
# TODO
# - integrate AlphaGo inspired model to play against

# References:
# https://github.com/aigagror/GymGo

import gym
import argparse
import numpy as np

from gym_hnef import hnef_vars

parser = argparse.ArgumentParser(description='Hnefatafl')
parser.add_argument('--mode', type=str, default='terminal')
parser.add_argument('--rules', type=str, default='historical')
parser.add_argument('--side', type=str, default='attacker')
args = parser.parse_args()

# create gym environment
hnef_env = gym.make('gym_hnef:hnef-v0', rule_set=args.rules, render_mode=args.mode)

done = False    # termination flag
# start main game loop
while not done:   
    if args.mode == 'gui':
        print(hnef_env.state[0] + hnef_env.state[1])
        hnef_env.render(mode="human")
        
        # player 1
        move = input("Type move =>\nSource 'row,col': ").split(',')
        src = int(move[0]), int(move[1])
        move = input("Destination 'row,col': ").split(',')
        dest = int(move[0]), int(move[1])
        action = tuple(src, dest)
        state, reward, done, info = hnef_env.step(action)

        if hnef_env.is_over():
            break

        # player 2 / RL model / random action
        action = hnef_env.random_action()
        state, reward, done, info = hnef_env.step(action)

    elif args.mode == 'terminal':
        if args.side == 'attacker':
            hnef_env.render(mode="terminal")
            # take input from terminal
            move = input("Type move =>\nSource 'row,col': ").split(',')
            src = int(move[0]), int(move[1])
            move = input("Destination 'row,col': ").split(',')
            dest = int(move[0]), int(move[1])
            action = src, dest
            state, reward, done, info = hnef_env.step(action)
        
            if hnef_env.is_over():
                print(">>Game Over<<")
                break

            # RL model / random action
            action = hnef_env.random_action()
            state, reward, done, info = hnef_env.step(action)

            if hnef_env.is_over():
                print(">>Game Over<<")
                break
        else:
            # RL model / random action
            action = hnef_env.random_action()
            state, reward, done, info = hnef_env.step(action)

            hnef_env.render(mode="terminal")

            if hnef_env.is_over():
                print(">>Game Over<<")
                break

            # take input from terminal
            move = input("Type move =>\nSource 'row,col': ").split(',')
            src = int(move[0]), int(move[1])
            move = input("Destination 'row,col': ").split(',')
            dest = int(move[0]), int(move[1])
            action = src, dest
            state, reward, done, info = hnef_env.step(action)

            if hnef_env.is_over():
                print(">>Game Over<<")
                break 
    elif args.mode == 'simulate':
        # RL model / random action
        action = hnef_env.random_action()
        state, reward, done, info = hnef_env.step(action)

        if hnef_env.is_over():
            print(">>Game Finished<<")
            print("The Attacker won!")
            print("Total of ", np.max(hnef_env.state[hnef_vars.TIME_CHNL]), " turns")
            break

        # RL model / random action
        action = hnef_env.random_action()
        state, reward, done, info = hnef_env.step(action)

        if hnef_env.is_over():
            print(">>Game Finished<<")
            print("The Defender won!")
            print("Total of ", np.max(hnef_env.state[hnef_vars.TIME_CHNL]), " turns")
            break
    else: 
        print("*** Invalid game mode -> Valid game modes include 'gui', 'terminal', and 'simulate'")
        break

if args.mode == 'gui':
    hnef_env.render(mode="human")
elif args.mode == 'terminal':
    hnef_env.render(mode="terminal")
elif args.mode == 'simulate':
    hnef_env.render(mode="terminal")
else: 
    print("*** Invalid game mode -> Valid game modes include 'gui', 'terminal', and 'simulate'")
