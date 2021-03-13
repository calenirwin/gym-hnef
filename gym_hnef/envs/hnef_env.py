# Written By: Tindur Sigurdason & Calen Irwin
# Written For: CISC-856 W21 (Reinforcement Learning) at Queen's U
# Last Modified Date: 2021-03-06 [CI]
# Purpose: 

# References:
# https://github.com/slowen/hnefatafl/blob/master/hnefatafl.py
# https://github.com/aigagror/GymGo

import gym

from hnef_gym import hnef_game, hnef_vars, rendering_helpers

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

    def render(self, mode='human'):
        if mode == 'terminal':
            print(self.__str__())
        elif mode == 'human':
            import pyglet
            from pyglet.window import mouse
            from pyglet.window import key

            screen = pyglet.canvas.get_display().get_default_screen()
            window = pyglet.window.Window(540, 540, style=window.Window.WINDOW_STYLE_TOOL, caption='Hnefatafl')

            # set a custom window icon --IF HAVE TIME--
            # icon2 = pyglet.image.load('32x32.png')
            # window.set_icon(icon1, icon2)

            self.window = window
            self.pyglet = pyglet
            self.user_action = None

            cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
            window.set_mouse_cursor(cursor)

            @window.event
            def on_draw():
                pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
                window.clear()

                pyglet.gl.glLineWidth(3)
                batch = pyglet.graphics.Batch()

                # draw the grid and labels
                rendering.draw_grid(batch, delta, self.size, lower_grid_coord, upper_grid_coord)

                # info on top of the board
                rendering.draw_info(batch, window_width, window_height, upper_grid_coord, self.state_)

                # Inform user what they can do
                rendering.draw_command_labels(batch, window_width, window_height)

                rendering.draw_title(batch, window_width, window_height)

                batch.draw()

                # draw the pieces
                rendering.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.state_)

             @window.event
            def on_mouse_press(x, y, button, modifiers):
                if button == mouse.LEFT:
                    grid_x = (x - lower_grid_coord)
                    grid_y = (y - lower_grid_coord)
                    x_coord = round(grid_x / delta)
                    y_coord = round(grid_y / delta)
                    try:
                        self.window.close()
                        pyglet.app.exit()
                        self.user_action = (x_coord, y_coord)
                    except:
                        pass

             @window.event
            def on_key_press(symbol, modifiers):
                if symbol == key.R:
                    self.reset()
                    self.window.close()
                    pyglet.app.exit()
                elif symbol == key.Q:
                    self.window.close()
                    pyglet.app.exit()
                    self.user_action = -1

            pyglet.app.run()

            return self.user_action        