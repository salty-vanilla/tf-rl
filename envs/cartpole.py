from gym.envs.classic_control.cartpole import CartPoleEnv as Base
import numpy as np 
import tensorflow as tf


# height, width
TARGET_SIZE = (40, 90)


class CartPoleEnv(Base):
    def __init__(self, state_mode='feature',
                 *args,
                 **kwargs):
        super().__init__()
        
        if state_mode not in ['feature', 'image']:
            raise ValueError
        self.state_mode = state_mode
    
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

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = tf.constant(screen)
        screen = tf.image.resize(screen, TARGET_SIZE)
        return screen

    def get_state_shape(self):
        if self.state_mode == 'image':
            return (*TARGET_SIZE, 3)
        else:
            return (4, )
    
    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done:
            reward = -1.

        if self.state_mode == 'image':
            current_screen = self.get_screen()
            last_screen = self.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            return next_state, reward, done, info
        else:
           return observation, reward, done, info

    def reset(self):
        if self.state_mode == 'image':
            super().reset()
            current_screen = self.get_screen()
            last_screen = self.get_screen()
            return current_screen - last_screen
        else:
           return super().reset()