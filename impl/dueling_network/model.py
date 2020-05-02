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
        self.feature = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            DenseBlock(128, activation_='relu', normalization=None),
        ])
        self.value_branch = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            DenseBlock(1),
        ])
        self.advantage_brach = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            DenseBlock(n_actions),
        ])
    
    @tf.function
    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        x = self.feature(x, training=training)
        v = self.value_branch(x, training=training)
        a = self.advantage_brach(x, training=training)
        return v + (a - tf.stop_gradient(tf.reduce_mean(a, axis=1, keepdims=True))) 


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
            tf.keras.layers.Flatten()
        ])
        self.value_branch = tf.keras.Sequential([
            DenseBlock(512, activation_='relu', normalization='batch'),
            DenseBlock(1),
        ])
        self.advantage_brach = tf.keras.Sequential([
            DenseBlock(512, activation_='relu', normalization='batch'),
            DenseBlock(n_actions),
        ])
    
    @tf.function
    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        x = self.feature(x, training=training)
        v = self.value_branch(x, training=training)
        a = self.advantage_brach(x, training=training)
        return v + (a - tf.stop_gradient(tf.reduce_mean(a, axis=1, keepdims=True))) 
