# Written By: Tindur Sigurdason & Calen Irwin
# Written For: CISC-856 W21 (Reinforcement Learning) at Queen's U
# Last Modified Date: 2021-03-06 [CI]
# Purpose: 

# References:
# https://github.com/slowen/hnefatafl/blob/master/hnefatafl.py
# https://github.com/aigagror/GymGo

import gym
import numpy as np
import random
import pyglet
from hnef_gym import hnef_game, hnef_vars, rendering_helpers

class HnefEnv(gym.Env):
    metadata = {'render.modes': ['terminal', 'gui']}
    hnef_vars = hnef_vars
    hnef_game = hnef_game

    def __init__(self, rule_set, render_mode, reward_method):
        self.rule_set = rule_set
        self.render_mode = render_mode
        self.reward_method = reward_method
        self.all_states = list()

        if rule_set.lower() == 'historical':
            self.size = 9
            self.state = hnef_game.init_state('historical')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 9, 9))
            self.action_space = gym.spaces.Discrete(hnef_game.action_size(self.state))
            
        else:
            self.size = 11
            self.state = hnef_game.init_state('copenhagen')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 11, 11))
            self.action_space = gym.spaces.Discrete(hnef_game.action_size(self.state))
            
        self.done = False

    def reset(self):
        self.state = hnef_game.init_state(self.rule_set)
        self.done = False
        return np.copy(self.state)

    # needs work
    # In: tuple of tuples action ((pos_x, pos_y), (new_pos_x, new_pos_y))
    # Out: observation (the new state), reward, done (True if game is finished, False otherwise), info (information of the state)
    def step(self, action):

        assert not self.done    # make sure that the game is not over
        
        self.all_states.append(self.state)  # keep track of all game states
        self.state = hnef_game.next_state(self.state, action)   # get next state
        self.done, winner = hnef_game.is_over(self.state, action)       # check if the game is over

        # check for repetition
        if len(self.all_states) > 6 and not self.done:
            
            # get the full board positions over the last six moves
            this_last = self.all_states[-1][hnef_vars.ATTACKER] + self.all_states[-1][hnef_vars.DEFENDER]
            this_next = self.all_states[-3][hnef_vars.ATTACKER] + self.all_states[-3][hnef_vars.DEFENDER] 
            this_first = self.all_states[-5][hnef_vars.ATTACKER] +  self.all_states[-5][hnef_vars.DEFENDER]
            other_last = self.all_states[-2][hnef_vars.ATTACKER] + self.all_states[-2][hnef_vars.DEFENDER]
            other_next = self.all_states[-4][hnef_vars.ATTACKER] + self.all_states[-4][hnef_vars.DEFENDER]
            other_first = self.all_states[-6][hnef_vars.ATTACKER] + self.all_states[-6][hnef_vars.DEFENDER]

            if (this_last == this_next and this_last == this_first) and (other_last == other_next and other_last == other_first):
                self.state[hnef_vars.DONE_CHNL] = 1
                winner = hnef_game.turn(state)

        # time constraint of 150 moves per player
        if self.state[hnef_vars.TIME_CHNL] > 300:
            self.state[hnef_vars.DONE_CHNL] = 1
            winner = 2
        else:
            self.state[hnef_vars.TIME_CHNL] += 1

        return np.copy(self.state), self.reward(winner), self.done, self.info()

    def is_over(self):
        return self.done

    def turn(self):
        return hnef_game.turn(self.state)

    def compute_valid_moves(self):
        return hnef_game.compute_valid_moves(self.state)

    def random_action(self):
        valid_moves = self.compute_valid_moves()
        return valid_moves[random.randrange(len(valid_moves))]  # randrange gets a random int between 0 and argument

    def info(self):
        return {
            'turn' : hnef_game.turn(self.state),
        }

    def state(self):
        return np.copy(self.state)

    def reward(self, winner):
        if not self[hnef_vars.DONE_CHNL]:
            return 0
        else:
            turn = hnef_game.turn(state)
            if turn == winner:
                return 1
            else:
                return 0

    def __str__(self):
        return hnef_game.str(self.state)

    def close(self):
        if hasattr(self, 'window'):
            assert hasattr(self, 'pyglet')
            self.window.close()
            self.pyglet.app.exit()

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
                rendering.draw_info(batch, window_width, window_height, upper_grid_coord, self.state)

                # Inform user what they can do
                rendering.draw_command_labels(batch, window_width, window_height)

                rendering.draw_title(batch, window_width, window_height)

                batch.draw()

                # draw the pieces
                rendering.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.state)

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