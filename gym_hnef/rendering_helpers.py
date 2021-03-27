import numpy as np
import pyglet

from gym_hnef import hnef_vars, hnef_game

#from gym_hnef inmport hnef_vars, hnef_game

def draw_circle(x, y, color, radius):
    num_sides = 50
    verts = [x, y]
    colors = list(color)
    for i in range(num_sides + 1):
        verts.append(x + radius * np.cos(i * np.pi * 2 / num_sides))
        verts.append(y + radius * np.sin(i * np.pi * 2 / num_sides))
        colors.extend(color)
    pyglet.graphics.draw(len(verts) // 2, pyglet.gl.GL_TRIANGLE_FAN,
                         ('v2f', verts), ('c3f', colors))


def draw_command_labels(batch, window_width, window_height):
    pyglet.text.Label('Resart (r) | Quit (q)',
                      font_name='Helvetica',
                      font_size=11,
                      x=20, y=window_height - 20, anchor_y='top', batch=batch, multiline=True, width=window_width)


def draw_info(batch, window_width, window_height, upper_grid_coord, state):
    turn = hnef_game.turn(state)
    turn_str = 'Attacker' if turn == hnef_vars.ATTACKER else 'Defender'
    # game_ended = hnef_game.game_ended(state)
    info_label = "Turn: {}".format(turn_str)

    pyglet.text.Label(info_label, font_name='Helvetica', font_size=11, x=window_width - 20, y=window_height - 20,
                      anchor_x='right', anchor_y='top', color=(0, 0, 0, 192), batch=batch, width=window_width / 2,
                      align='right', multiline=True)


def draw_title(batch, window_width, window_height):
    pyglet.text.Label("Hnefatafl", font_name='Helvetica', font_size=20, bold=True, x=window_width / 2, y=window_height - 20,
                      anchor_x='center', anchor_y='top', color=(0, 0, 0, 255), batch=batch, width=window_width / 2,
                      align='center')


def draw_grid(batch, delta, board_size, lower_grid_coord, upper_grid_coord):
    label_offset = 20
    left_coord = lower_grid_coord
    right_coord = lower_grid_coord
    ver_list = []
    color_list = []
    num_vert = 0
    for i in range(board_size):
        # horizontal
        ver_list.extend((lower_grid_coord, left_coord,
                         upper_grid_coord, right_coord))
        # vertical
        ver_list.extend((left_coord, lower_grid_coord,
                         right_coord, upper_grid_coord))
        color_list.extend([0.3, 0.3, 0.3] * 4)  # black
        # label on the left
        pyglet.text.Label(str(i),
                          font_name='Courier', font_size=11,
                          x=lower_grid_coord - label_offset, y=left_coord,
                          anchor_x='center', anchor_y='center',
                          color=(0, 0, 0, 255), batch=batch)
        # label on the bottom
        pyglet.text.Label(str(i),
                          font_name='Courier', font_size=11,
                          x=left_coord, y=lower_grid_coord - label_offset,
                          anchor_x='center', anchor_y='center',
                          color=(0, 0, 0, 255), batch=batch)
        left_coord += delta
        right_coord += delta
        num_vert += 4
    batch.add(num_vert, pyglet.gl.GL_LINES, None,
              ('v2f/static', ver_list), ('c3f/static', color_list))


def draw_pieces(batch, lower_grid_coord, delta, piece_r, size, state):
    for i in range(size):
        for j in range(size):
            # ATTACKER
            if state[hnef_vars.ATTACKER, i, j] == 1:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.05882352963, 0.180392161, 0.2470588237],
                            piece_r)  # 0 for black
            # DEFENDER
            if state[hnef_vars.DEFENDER, i, j] == 1:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.9754120272] * 3, piece_r)  # 255 for white
            
            if state[hnef_vars.DEFENDER, i, j] == 2:
                draw_circle(lower_grid_coord + i * delta, lower_grid_coord + j * delta,
                            [0.04, 0.1, 0.9], piece_r)  # random color