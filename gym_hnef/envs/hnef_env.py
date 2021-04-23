# Written By: Tindur Sigurdason & Calen Irwin
# Written For: CISC-856 W21 (Reinforcement Learning) at Queen's U

# References:
# https://github.com/slowen/hnefatafl/blob/master/hnefatafl.py
# https://github.com/aigagror/GymGo

import gym
import numpy as np
import random
import pyglet

from gym_hnef import hnef_game, hnef_vars, rendering_helpers

# custom gym Env object built to play Hnefatafl (aka Viking Chess)
# contains important information about the game environment
# and basic RL environment methods needed to implement more complex AI models
class HnefEnv(gym.Env):
    metadata = {'render.modes': ['terminal', 'gui']}
    hnef_vars = hnef_vars
    hnef_game = hnef_game

    def __init__(self, rule_set, render_mode):
        self.rule_set = rule_set
        self.render_mode = render_mode
        self.all_states = list()
        self.all_actions = list()

        if rule_set.lower() == 'historical':
            self.size = 9
            self.state = hnef_game.init_state('historical')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 9, 9))
            self.action_space = gym.spaces.Discrete(6561)

        elif rule_set.lower() == 'mini':
            self.size = 5
            self.state = hnef_game.init_state('mini')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 5, 5))
            self.action_space = gym.spaces.Discrete(625)
            
        else:
            self.size = 11
            self.state = hnef_game.init_state('copenhagen')
            self.observation_space = gym.spaces.Box(np.float32(0), np.float32(hnef_vars.NUM_CHNLS), shape=(hnef_vars.NUM_CHNLS, 11, 11))
            self.action_space = gym.spaces.Discrete(14641)
            
        self.done = False

    # Method to reset the game state to its initial position and reset the done flag
    def reset(self):
        self.state = hnef_game.init_state(self.rule_set)
        self.done = False
        return np.copy(self.state)

    # In: tuple of tuples action ((pos_x, pos_y), (new_pos_x, new_pos_y))
    # Out: observation (the new state), reward, done (True if game is finished, False otherwise), info (information of the state)
    def step(self, action):

        assert not self.done    # make sure that the game is not over
        
        self.all_states.append(self.state)  # keep track of all game states
        self.all_actions.append(action)  # keep track of all actions taken
        self.state = hnef_game.next_state(self.state, action)   # get next state
        self.done, winner = hnef_game.is_over(self.state, action)       # check if the game is over

        # check for repetition
        if len(self.all_actions) > 6 and not self.done:
            
            # get the full board positions over the last six moves
            this_last = self.all_actions[-1]
            this_next = self.all_actions[-3]
            this_first = self.all_actions[-5]
            other_last = self.all_actions[-2]
            other_next = self.all_actions[-4]
            other_first = self.all_actions[-6]

            if (np.mean(this_last == this_next) == 1 and np.mean(this_last == this_first) == 1) and (np.mean(other_last == other_next) == 1 and np.mean(other_last == other_first) == 1):
                print("***Repitition condition met")
                self.done = True
                winner = hnef_game.turn(self.state)

        # time constraint of 150 moves per player
        if np.max(self.state[hnef_vars.TIME_CHNL]) > 300:
            self.done = True
            winner = 2
            print("***Exceeded time limit")
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
        if not self.done:
            return 0
        else:
            current_player = hnef_game.turn(self.state)
            if current_player == winner:
                return 1
            elif winner == 2:   # game ended as a draw
                return 2
            elif winner == -1:
                print('***Invalid winner')
                assert False
            else:
                return 0

    def __str__(self):
        return hnef_game.str(self.state)

    def close(self):
        if hasattr(self, 'window'):
            assert hasattr(self, 'pyglet')
            self.window.close()
            self.pyglet.app.exit()

    # used to render GUI
    # NEEDS WORK
    def render(self, mode='human'):
        if mode == 'terminal':
            print(self.__str__())
        elif mode == 'human':
            import pyglet
            from pyglet.window import mouse
            from pyglet.window import key

            screen = pyglet.canvas.get_display().get_default_screen()
            window_width = int(min(screen.width, screen.height) * 2 / 3)
            window_height = int(window_width * 1.2)
            window = pyglet.window.Window(window_width, window_height, caption='Hnefatafl')

            # set a custom window icon --IF HAVE TIME--
            # icon2 = pyglet.image.load('32x32.png')
            # window.set_icon(icon1, icon2)

            self.window = window
            self.pyglet = pyglet
            self.user_action = None

            cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
            window.set_mouse_cursor(cursor)

            lower_grid_coord = window_width * 0.075
            board_size = window_width * 0.85
            upper_grid_coord = board_size + lower_grid_coord
            delta = board_size / (self.size - 1)
            piece_r = delta / 3.3  # radius

            @window.event
            def on_draw():
                pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
                window.clear()

                pyglet.gl.glLineWidth(3)
                batch = pyglet.graphics.Batch()

                # draw the grid and labels
                rendering_helpers.draw_grid(batch, delta, self.size, lower_grid_coord, upper_grid_coord)

                # info on top of the board
                rendering_helpers.draw_info(batch, window_width, window_height, upper_grid_coord, self.state)

                # Inform user what they can do
                rendering_helpers.draw_command_labels(batch, window_width, window_height)

                rendering_helpers.draw_title(batch, window_width, window_height)

                batch.draw()

                # draw the pieces
                rendering_helpers.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.state)
                
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