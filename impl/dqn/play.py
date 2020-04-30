import tensorflow as tf
import numpy as np
import yaml
import sys
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, ImageMagickWriter, FFMpegWriter
from cartpole import env
from model import Model


def display_frames_as_gif(dst_path, frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1]//10, frames[0].shape[0]//10),
               dpi=100)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    
    dst_path = Path(dst_path)
    ext = dst_path.suffix[1:]
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if ext == 'gif':
        writer = ImageMagickWriter()
    elif ext == 'mp4':
        writer = FFMpegWriter()
    else:
        raise ValueError

    anim = FuncAnimation(plt.gcf(), animate, frames=len(frames),
                         interval=50)
    anim.save(str(dst_path), writer=writer)

yml_path = sys.argv[1]
with open(yml_path) as f:
    config = yaml.load(f)
logdir = Path(config['logdir'])

model = Model(env.action_space.n)
model.build((None, *env.get_screen_shape()))
model.load_weights(str(logdir / 'model' / f'model_{config["test_episode"]}.h5'))

env.reset()
last_screen = env.get_screen()
current_screen = env.get_screen()
state = current_screen - last_screen
frames = [current_screen]

rewards = 0
for t in count():
    action = tf.argmax(model(tf.expand_dims(state, 0)), axis=1)
    action = tf.reshape(action, ())
    action = np.array(action)
    _, reward, done, _ = env.step(action)

    last_screen = current_screen
    current_screen = env.get_screen()

    frames.append(current_screen)
    rewards += reward

    if not done:
        next_state = current_screen - last_screen
    else:
        next_state = None
        break

    state = next_state

display_frames_as_gif(logdir / 'play.gif', frames)
display_frames_as_gif(logdir / 'play.mp4', frames)