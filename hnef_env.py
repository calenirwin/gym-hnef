import gym
import pyglet

# https://github.com/slowen/hnefatafl/blob/master/hnefatafl.py
class HnefEnv(gym.Env):
    def __init__(self, env_config={}):
        self.observation_space = <gym.space>
        self.action_space = <gym.space>

        def reset(self):
            return observation_space

        def step(self, action):
            return observation, reward, done, info


        def render(self):
            window = pyglet.window.Window(540, 540, style=window.Window.WINDOW_STYLE_TOOL, caption='Hnefatafl')

            # load window icon
            # icon2 = pyglet.image.load('32x32.png')
            # window.set_icon(icon1, icon2)
            cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
            window.set_mouse_cursor(cursor)


            