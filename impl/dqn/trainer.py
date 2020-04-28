import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import random
from itertools import count
import time
from replay_memory import ReplayMemory, Transition
from model import Model


class Trainer:
    def __init__(self, input_shape,
                 env,
                 n_actions,
                 memory,
                 gamma=0.999,
                 eps_start=0.99,
                 eps_end=0.05,
                 eps_decay=200):
        self.env = env
        self.n_actions = n_actions
        self.memory = memory
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # self.optimizer = tf.keras.optimizers.RMSprop()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.policy_model = Model(env.action_space.n)
        self.policy_model.build((None, *input_shape))
        self.target_model = Model(env.action_space.n)
        self.target_model.build((None, *input_shape))

        self.steps_done = 0

        self.target_model.set_weights(self.policy_model.get_weights())

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = tf.summary.create_file_writer("logs/" + current_time)

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
        transitions = self.memory.sample(batch_size)
        
        batch = Transition(*zip(*transitions))
        
        state_batch = tf.concat([batch.state], axis=0)
        action_batch = tf.concat([batch.action], axis=0)
        reward_batch = tf.concat([batch.reward], axis=0)

        # @tf.function
        def _update(state_batch, 
                    action_batch,
                    reward_batch):
            loss_fn = tf.keras.losses.Huber()

            non_final_mask = tf.constant(tuple(map(lambda s: s is not None,
                                                batch.next_state)))
            non_final_next_states = tf.concat([[s if s is not None else tf.zeros_like(state_batch[0]) 
                                                for s in batch.next_state]],
                                            axis=0)
        
            with tf.GradientTape() as tape:           
                state_action_values = tf.reduce_sum(
                    self.policy_model(state_batch, training=True) * tf.one_hot(action_batch, self.n_actions),
                    axis=1
                )

                next_state_values = tf.zeros(shape=(batch_size, ),
                                             dtype=tf.float32)
                preds = tf.reduce_max(self.target_model(non_final_next_states,
                                                        training=False),
                                      axis=-1)
                next_state_values = tf.where(non_final_mask, preds, next_state_values)
                expected_state_action_values = (next_state_values * self.gamma) + reward_batch

                loss = tf.reduce_mean(loss_fn(state_action_values, expected_state_action_values))

            grads = tape.gradient(loss, self.policy_model.trainable_variables)
            grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))
            return loss

        return np.array(_update(state_batch, action_batch, reward_batch))


    def fit(self, batch_size=32,
            episodes=100,
            target_update=10):
        for episode in range(1, episodes + 1):
            print(f'\nepisode {episode} / {episodes}')
            start = time.time()

            self.env.reset()
            last_screen = self.env.get_screen()
            current_screen = self.env.get_screen()
            state = current_screen - last_screen

            rewards = 0
            losses = 0
            for t in count():
                action = self.select_action(tf.expand_dims(state, 0))
                action = np.array(action)
                _, reward, done, _ = self.env.step(action)
                reward = tf.constant(reward, dtype=tf.float32)
        
                last_screen = current_screen
                current_screen = self.env.get_screen()

                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                
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
        print()