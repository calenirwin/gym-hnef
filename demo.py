import argparse
import gym

parser = argparse.ArgumentParser(description='Hnefatafl')
parser.add_argument('--mode', type=str, default='gui')
parser.add_argument('--rules', type=str, default='historical')
args = parser.parse_args()

# create gym environment
hnef_env = gym.make('gym_hnef:hnef-v0')

done = False    # termination flag
# start main game loop
while not done:
    if args.mode == 'gui':
        hnef_env.render(mode="human")
        # player 1
        # move = first click on piece that we want to move then valid move location
        # action = coordinates of piece we want to move and coordinates of piece
        state, reward, done, info = go_env.step(action)

        if hnef_env.game_ended():
            break

        # player 2 / RL model / random action
        action = hnef_env.uniform_random_action()
        state, reward, done, info = hnef_env.step(action)

    else if args.mode == 'terminal':
        hnef_env.render(mode="terminal")
        # action = RL model picks piece and a valid location to move to (tuple)
        state, reward, done, info = go_env.step(action)

        if hnef_env.game_ended():
            break

        # RL model / random action
        action = hnef_env.uniform_random_action()
        state, reward, done, info = hnef_env.step(action)
    else: 
        print("*** Invalid game mode -> Valid game modes include 'gui' and 'terminal'")
if args.mode == 'gui':
    hnef_env.render(mode="human")
else if args.mode == 'terminal'
    hnef_env.render(mode="terminal")
else: 
    print("*** Invalid game mode -> Valid game modes include 'gui' and 'terminal'")
