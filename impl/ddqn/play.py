import tensorflow as tf
import numpy as np
import yaml
import sys
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, ImageMagickWriter, FFMpegWriter
sys.path.append('../../')
from envs import CartPoleEnv
from model import CNN, MLP


def display_frames_as_gif(dst_path, frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1]//100, frames[0].shape[0]//100))
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

env = CartPoleEnv(**config['env_params'])

if config['env_params']['state_mode'] == 'image':
    model = CNN(env.action_space.n)
else:
    model = MLP(env.action_space.n)

model.build((None, *env.get_state_shape()))
model.load_weights(str(logdir / 'model' / f'model_{config["test_episode"]}.h5'))

env.reset()

state = env.reset()
frames = [env.render(mode='rgb_array')]

rewards = 0
for t in count():
    state = tf.constant(state, dtype=tf.float32)
    action = tf.argmax(model(tf.expand_dims(state, 0)), axis=1)
    action = tf.reshape(action, ())
    action = np.array(action)
    next_state, reward, done, _ = env.step(action)

    frames.append(env.render(mode='rgb_array'))
    rewards += reward

    if done:
        break
    state = next_state

display_frames_as_gif(logdir / 'play.gif', frames)
display_frames_as_gif(logdir / 'play.mp4', frames)