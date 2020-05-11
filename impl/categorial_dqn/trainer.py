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
                 min_q_value=0.,
                 max_q_value=20.0,
                 n_bins=51,
                 eps_start=0.99,
                 eps_end=0.05,
                 eps_decay=200,
                 logdir=None):
        self.env = env
        self.n_actions = n_actions
        self.memory = memory
        self.gamma = gamma
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value
        self.n_bins = n_bins
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # self.optimizer = tf.keras.optimizers.RMSprop()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.policy_model = policy_model
        self.policy_model(tf.random.normal(shape=(1, *input_shape), dtype=tf.float32))
        self.target_model = target_model
        self.target_model(tf.random.normal(shape=(1, *input_shape), dtype=tf.float32))

        self.steps_done = 0

        self.target_model.set_weights(self.policy_model.get_weights())

        self.support = tf.linspace(min_q_value, max_q_value, n_bins)
        self.delta_z = (max_q_value-min_q_value) / (n_bins - 1)
        self.m = None
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = Path(logdir) if logdir else Path(f'logs/{current_time}')
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(str(self.logdir))

    def select_action(self, state):
        p = self.policy_model(state, training=False)
        z = tf.reshape(self.support, (1, 1, self.n_bins))
        state_action_value = tf.reduce_sum(p*z, axis=2)
        action = tf.argmax(state_action_value, axis=1)
        return tf.reshape(action, ())

    def update_model(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.
        transitions, indices, weights = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        weights = tf.constant(weights, dtype=tf.float32)

        state_batch = tf.concat([batch.state], axis=0)
        action_batch = tf.concat([batch.action], axis=0)
        reward_batch = tf.concat([batch.reward], axis=0)

        self.policy_model.reset_noise()
        self.target_model.reset_noise()
        # @tf.function
        def _update(state_batch, 
                    action_batch,
                    reward_batch):
            non_final_mask = tf.constant(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                         dtype=tf.bool)
            non_final_next_states = tf.concat([[s if s is not None else tf.zeros_like(state_batch[0]) 
                                                for s in batch.next_state]],
                                            axis=0)
            next_state_p_ = self.target_model(non_final_next_states, training=False)
            z = tf.reshape(self.support, (1, 1, self.n_bins))
            next_state_actions = tf.argmax(tf.reduce_sum(next_state_p_*z, axis=2), axis=1)
            next_state_p_ = tf.reduce_sum(
                next_state_p_ * tf.expand_dims(tf.one_hot(next_state_actions, self.n_actions), axis=-1),
                axis=1
            )
            next_state_p = tf.zeros(shape=(batch_size, self.n_bins),
                                    dtype=tf.float32)
            next_state_p = tf.where(tf.expand_dims(non_final_mask, axis=1), next_state_p_, next_state_p)
            gamma = tf.where(non_final_mask, 
                             tf.constant(self.gamma, dtype=tf.float32), 
                             tf.constant(0., dtype=tf.float32))

            target = tf.tile(tf.expand_dims(reward_batch, axis=1), (1, self.n_bins))
            target += tf.expand_dims(gamma, axis=1) * tf.expand_dims(self.support, axis=0)
            target = tf.clip_by_value(target, self.min_q_value, self.max_q_value)
            b = (target - self.min_q_value) / self.delta_z
            l = tf.math.floor(b)
            u = tf.math.ceil(b)
            u_id = tf.cast(u, tf.int32)
            l_id = tf.cast(l, tf.int32)

            index_help = tf.tile(tf.reshape(tf.range(batch_size),[-1, 1]), tf.constant([1, self.n_bins]))
            index_help = tf.expand_dims(index_help, -1)
            u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=2)
            l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=2)

            with tf.GradientTape() as tape:                   
                current_state_p = tf.reduce_sum(
                    self.policy_model(state_batch, training=True) * tf.expand_dims(tf.one_hot(action_batch, self.n_actions), axis=-1),
                    axis=1
                )
                loss_batch = next_state_p*(u-b) * tf.math.log(tf.gather_nd(current_state_p, l_id)+tf.keras.backend.epsilon()) \
                            + next_state_p*(b-l) * tf.math.log(tf.gather_nd(current_state_p, u_id)+tf.keras.backend.epsilon()) 
                loss_batch = -tf.reduce_sum(loss_batch, axis=1)
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
        self.m = tf.Variable(tf.zeros(shape=(batch_size, self.n_bins), dtype=tf.float32))
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