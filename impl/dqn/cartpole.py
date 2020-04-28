import gym
import numpy as np
import tensorflow as tf
from types import MethodType

# height, width
TARGET_SIZE = (40, 90)

def get_cart_location(self, screen_width):
    world_width = self.x_threshold*2
    scale = screen_width / world_width
    return int(self.state[0]*scale + screen_width / 2.)

def get_screen(self):
    screen = self.render(mode='rgb_array')
    screen_height, screen_width, _ = screen.shape
    screen = screen[int(screen_height*0.4):int(screen_height * 0.8), :, :]
    # Cart is in the lower half, so strip off the top and bottom of the screen
    view_width = int(screen_width * 0.6)
    cart_location = self.get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, slice_range, :]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = tf.constant(screen)
    screen = tf.image.resize(screen, TARGET_SIZE)
    return screen

def get_screen_shape(self):
    screen = self.render(mode='rgb_array')
    screen_height, screen_width, _ = screen.shape
    return (*TARGET_SIZE, 3)

env = gym.make('CartPole-v0').unwrapped
env.reset()
env.get_cart_location = MethodType(get_cart_location, env)
env.get_screen = MethodType(get_screen, env) 
env.get_screen_shape = MethodType(get_screen_shape, env)
