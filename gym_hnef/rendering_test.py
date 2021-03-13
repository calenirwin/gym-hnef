import pyglet
from pyglet.window import mouse
from pyglet.window import key

import rendering_helpers

screen = pyglet.canvas.get_display().get_default_screen()
window_width = int(min(screen.width, screen.height) * 2 / 3)
window_height = int(window_width * 1.2)
window = pyglet.window.Window(window_width, window_height)

# set a custom window icon --IF HAVE TIME--
# icon2 = pyglet.image.load('32x32.png')
# window.set_icon(icon1, icon2)



cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
window.set_mouse_cursor(cursor)

@window.event
def on_draw():
    pyglet.gl.glClearColor(0.7, 0.5, 0.3, 1)
    window.clear()

    pyglet.gl.glLineWidth(3)
    batch = pyglet.graphics.Batch()

    # draw the grid and labels
    # rendering_helpers.draw_grid(batch, delta, self.size, lower_grid_coord, upper_grid_coord)

    # info on top of the board
    #rendering_helpers.draw_info(batch, window_width, window_height, upper_grid_coord, self.state_)

    # Inform user what they can do
    rendering_helpers.draw_command_labels(batch, window_width, window_height)

    rendering_helpers.draw_title(batch, window_width, window_height)

    batch.draw()

    # draw the pieces
    #rendering_helpers.draw_pieces(batch, lower_grid_coord, delta, piece_r, self.size, self.state_)

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        grid_x = (x - lower_grid_coord)
        grid_y = (y - lower_grid_coord)
        x_coord = round(grid_x / delta)
        y_coord = round(grid_y / delta)
        try:
            window.close()
            pyglet.app.exit()
        except:
            pass

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.R:
        swindow.close()
        pyglet.app.exit()
    elif symbol == key.Q:
        window.close()
        pyglet.app.exit()
        user_action = -1

pyglet.app.run()