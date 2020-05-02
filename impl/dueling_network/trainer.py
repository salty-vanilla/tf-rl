import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import random
import copy
from pathlib import Path
from itertools import count
import time
import sys
sys.path.append('../../')
from replay_memory import Transition


class Trainer:
    def __init__(self, input_shape,
                 policy_model,
                 target_model,
                 env,
                 n_actions,
                 memory,
                 gamma=0.999,
                 eps_start=0.99,
                 eps_end=0.05,
                 eps_decay=200,
                 logdir=None):
        self.env = env
        self.n_actions = n_actions
        self.memory = memory
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # self.optimizer = tf.keras.optimizers.RMSprop()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.policy_model = policy_model
        self.policy_model.build((None, *input_shape))
        self.target_model = target_model
        self.target_model.build((None, *input_shape))

        self.steps_done = 0

        self.target_model.set_weights(self.policy_model.get_weights())

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = Path(logdir) if logdir else Path(f'logs/{current_time}')
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(str(self.logdir))

    def select_action(self, state):
        sample = np.random.rand()
        eps_threshold = self.eps_end + \
            (self.eps_start-self.eps_end)*np.exp(-self.steps_done/self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return tf.reshape(
                tf.argmax(self.policy_model(state, training=False), axis=1),
                ()
            )
        else:
            return tf.random.uniform(shape=(),
                                     minval=0,
                                     maxval=self.n_actions,
                                     dtype=tf.int32)

    def update_model(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.
        transitions, indices, weights = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        weights = tf.constant(weights, dtype=tf.float32)

        state_batch = tf.concat([batch.state], axis=0)
        action_batch = tf.concat([batch.action], axis=0)
        reward_batch = tf.concat([batch.reward], axis=0)

        # @tf.function
        def _update(state_batch, 
                    action_batch,
                    reward_batch):
            loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

            non_final_mask = tf.constant(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                         dtype=tf.bool)
            non_final_next_states = tf.concat([[s if s is not None else tf.zeros_like(state_batch[0]) 
                                                for s in batch.next_state]],
                                            axis=0)

            next_state_values = tf.zeros(shape=(batch_size, ),
                                         dtype=tf.float32)
            next_state_actions = tf.argmax(self.policy_model(non_final_next_states), axis=1)

            next_state_values_ = tf.reduce_sum(
                self.target_model(non_final_next_states) * tf.one_hot(next_state_actions, self.n_actions),
                axis=1
            )
            next_state_values = tf.where(non_final_mask, next_state_values_, next_state_values)
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
            with tf.GradientTape() as tape:           
                state_action_values = tf.reduce_sum(
                    self.policy_model(state_batch, training=True) * tf.one_hot(action_batch, self.n_actions),
                    axis=1
                )

                loss_batch = loss_fn(state_action_values, expected_state_action_values)
                loss = tf.reduce_mean(loss_batch*weights)

            grads = tape.gradient(loss, self.policy_model.trainable_variables)
            grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
            return loss, loss_batch

        loss, loss_batch = _update(state_batch, action_batch, reward_batch)

        loss_batch = np.array(loss_batch)
        for index, error in zip(indices, loss_batch):
            self.memory.update(index, error)

        return np.array(loss)


    def fit(self, batch_size=32,
            episodes=100,
            target_update=10,
            save_steps=100):
        for episode in range(1, episodes + 1):
            print(f'\nepisode {episode} / {episodes}')
            start = time.time()

            state = self.env.reset()
            rewards = 0
            losses = 0
            for t in count():
                state = tf.constant(state, dtype=tf.float32)
                action = self.select_action(tf.expand_dims(state, 0))
                action = np.array(action)
                next_state, reward, done, _ = self.env.step(action)
                reward = tf.constant(reward, dtype=tf.float32)
        
                self.memory.push(state, action, next_state, reward)

                state = next_state

                loss = self.update_model(batch_size)
                
                if done:
                    break
                        
                elapsed = time.time() - start
                rewards += reward
                losses += loss
                print(f'iter : {t}  {elapsed:.1f}[s]  loss : {loss:.4f}  rewards : {rewards}', end='\r')
            
            losses /= t
            with self.writer.as_default():
                tf.summary.scalar('rewards', rewards, step=episode)
                tf.summary.scalar('loss', losses, step=episode)
            self.writer.flush()

            if episode % target_update == 0:
                self.target_model.set_weights(self.policy_model.get_weights())
            if episode % save_steps == 0:
                (self.logdir / 'model' ).mkdir(exist_ok=True)
                self.policy_model.save_weights(str(self.logdir / 'model' / f'model_{episode}.h5'))

        print()