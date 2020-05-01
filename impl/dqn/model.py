import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../../')
from ops.blocks import ConvBlock, DenseBlock


class MLP(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.denses = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            DenseBlock(128, activation_='relu', normalization=None),
            DenseBlock(self.n_actions),
        ])

    @tf.function
    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        return self.denses(x, training=training)


class CNN(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.feature = tf.keras.Sequential([
            ConvBlock(32, 
                      kernel_size=(9, 9),
                      activation_='relu',
                      sampling='stride',
                      normalization='batch'),

            ConvBlock(64, 
                      kernel_size=(5, 5),
                      activation_='relu',
                      sampling='same',
                      normalization='batch'),
            ConvBlock(64, 
                      kernel_size=(5, 5),
                      activation_='relu',
                      sampling='stride',
                      normalization='batch'),

            ConvBlock(128, 
                      kernel_size=(5, 5),
                      activation_='relu',
                      sampling='same',
                      normalization='batch'),
            ConvBlock(128, 
                      kernel_size=(5, 5),
                      activation_='relu',
                      sampling='stride',
                      normalization='batch'),
        ])
        self.dense1 = DenseBlock(512,
                                 activation_='relu',
                                 normalization='batch')
        self.dense2 = DenseBlock(n_actions)

    @tf.function
    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        x = self.feature(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        return x

