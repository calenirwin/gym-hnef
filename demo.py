import argparse
import gym

parser = argparse.ArgumentParser(description='Hnefatafl')
parser.add_argument('--mode', type=str, default='terminal')
parser.add_argument('--rules', type=str, default='historical')
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

        if hnef_env.is_over()[0]:
            break

        # player 2 / RL model / random action
        action = hnef_env.uniform_random_action()
        state, reward, done, info = hnef_env.step(action)

    elif args.mode == 'terminal':
        hnef_env.render(mode="terminal")

        # player 1
        move = input("Type move =>\nSource 'row,col': ").split(',')
        src = int(move[0]), int(move[1])
        move = input("Destination 'row,col': ").split(',')
        dest = int(move[0]), int(move[1])
        action = src, dest
        print(action)
        state, reward, done, info = hnef_env.step(action)


        if hnef_env.is_over()[0]:
            break

        # RL model / random action
        action = hnef_env.uniform_random_action()
        state, reward, done, info = hnef_env.step(action)
    else: 
        print("*** Invalid game mode -> Valid game modes include 'gui' and 'terminal'")
if args.mode == 'gui':
    hnef_env.render(mode="human")
elif args.mode == 'terminal':
    hnef_env.render(mode="terminal")
else: 
    print("*** Invalid game mode -> Valid game modes include 'gui' and 'terminal'")
