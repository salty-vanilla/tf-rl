import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../../')
from ops.blocks import ConvBlock, DenseBlock


class Model(tf.keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.conv1 = ConvBlock(16, 
                               kernel_size=(9, 9),
                               activation_='relu',
                               sampling='stride',
                               padding='valid',
                               normalization='batch')
        self.conv2 = ConvBlock(32, 
                               kernel_size=(5, 5),
                               activation_='relu',
                               sampling='stride',
                               padding='valid',
                               normalization='batch')
        self.conv3 = ConvBlock(32, 
                               kernel_size=(5, 5),
                               activation_='relu',
                               sampling='stride',
                               padding='valid',
                               normalization='batch')
        self.dense1 = DenseBlock(128,
                                 activation_='relu',
                                 normalization='batch')
        self.dense2 = DenseBlock(n_actions)

    @tf.function
    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        return x

