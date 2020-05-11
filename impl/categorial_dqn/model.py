import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../../')
from ops.blocks import ConvBlock, DenseBlock
from ops.layers import NoisyDense


class Base(tf.keras.Model):
    def reset_noise(self):
        def get_all_layers(layers):
            for l in layers:
                if hasattr(l, 'layers'):
                    for _l in l.layers:
                        if hasattr(_l, 'layers'):
                            _layers = [__l for __l in get_all_layers(_l.layers)]
                            yield _layers 
                        else:
                            yield _l
                else:
                    yield l

        layers = []
        for _layers in get_all_layers(self.layers):
            if isinstance(_layers, list):
                layers.extend(_layers)
            else:
                layers.append(_layers)
        
        for layer in layers:
            if hasattr(layer, 'reset_noise'):
                layer.reset_noise()


class MLP(Base):
    def __init__(self, n_actions, n_bins=51):
        super().__init__()
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.feature = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            DenseBlock(128, activation_='relu', normalization=None),
        ])
        self.value_branch = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            NoisyDense(1*n_bins),
        ])
        self.advantage_brach = tf.keras.Sequential([
            DenseBlock(128, activation_='relu', normalization=None),
            NoisyDense(n_actions*n_bins),
        ])
    
    @tf.function
    def call(self, inputs,
             training=None,
             mask=None,
             softmax=True):
        x = inputs
        x = self.feature(x, training=training)
        v = self.value_branch(x, training=training)
        a = self.advantage_brach(x, training=training)

        v = tf.reshape(v, (v.shape[0], 1, self.n_bins))
        a = tf.reshape(a, (a.shape[0], self.n_actions, self.n_bins))
        y = v + (a - tf.stop_gradient(tf.reduce_mean(a, axis=1, keepdims=True))) 
        if softmax:
            y = tf.nn.softmax(y, axis=2)
        return y

class CNN(Base):
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
            NoisyDense(1),
        ])
        self.advantage_brach = tf.keras.Sequential([
            DenseBlock(512, activation_='relu', normalization='batch'),
            NoisyDense(n_actions),
        ])
    
    @tf.function
    def call(self, inputs,
             training=None,
             mask=None,
             softmax=True):
        x = inputs
        x = self.feature(x, training=training)
        v = self.value_branch(x, training=training)
        a = self.advantage_brach(x, training=training)

        v = tf.reshape(v, (v.shape[0], 1, self.n_bins))
        a = tf.reshape(a, (a.shape[0], self.n_actions, self.n_bins))
        y = v + (a - tf.stop_gradient(tf.reduce_mean(a, axis=1, keepdims=True))) 
        if softmax:
            y = tf.nn.softmax(y, axis=2)
        return y