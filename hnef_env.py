# Written By: Tindur Sigurdason & Calen Irwin
# Written For: CISC-856 W21 (Reinforcement Learning) at Queen's U
# Last Modified Date: 2021-03-06 [CI]
# Purpose: 

# References:
# https://github.com/slowen/hnefatafl/blob/master/hnefatafl.py
# https://github.com/aigagror/GymGo

import gym
import pyglet

import hnef_game
import hnef_vars

class HnefEnv(gym.Env):

    def __init__(self, rule_set, render_mode, reward_method):
        self.rule_set = rule_set
        self.render_mode = render_mode
        self.reward_method = reward_method

        if rule_set.lower() == 'historical':
            self.state = hnef_game.init_state('historical')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 9, 9))
            self.action_space = gym.spaces.Discrete(hnef_game.action_size(self.state))
            
        else:
            self.state = hnef_game.init_state('copenhagen')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 11, 11))
            self.action_space = gym.spaces.Discrete(hnef_game.action_size(self.state))
            
        self.done = False

    # needs work
    def reset(self):
        return observation_space

    # needs work
    # In: tuple of tuples action ((pos_x, pos_y), (new_pos_x, new_pos_y))
    # Out: observation (the new state), reward, done (True if game is finished, False otherwise), info (information of the state)
    def step(self, action):

        # make sure that the game is still going
        assert not self.done        

        return observation, reward, done, info

    def winner(self):
        if self.game_ended():
            return self.winning()
        else:
            return 0

    def reward(self):
        return self.winner()


    def __str__(self):
        return gogame.str(self.state_)

    def render(self, mode='terminal'):
        if mode == 'terminal':
            print(self.__str__())
        elif mode == 'human':
            import pyglet
            from pyglet.window import mouse
            from pyglet.window import key
        window = pyglet.window.Window(540, 540, style=window.Window.WINDOW_STYLE_TOOL, caption='Hnefatafl')

        # load window icon
        # icon2 = pyglet.image.load('32x32.png')
        # window.set_icon(icon1, icon2)
        cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
        window.set_mouse_cursor(cursor)


            