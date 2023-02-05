from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
import numpy as np


class ResidualUnit(object):
    """Residual unit block (unidimensional)

    Args:
        n_samples_out: int
            Number of output samples.
        n_filters_out: int
            Number of output filters.
        kernel_initializer (optional): str
            Initializer for the weights matrices. See Keras initializers. By default, it uses 'he_normal'.
        dropout_keep_prob (optional): float [0, 1)
            Dropout rate used in all Dropout layers. Default is 0.8
        kernel_size (optional): int
            Kernel size for convolutional layers. Default is 17.
        preactivation (optional): bool
            When preactivation is true usefully preactivation architecture proposed in [1].
            Otherwise, use architecture suggested in the original ResNet paper [2]. By default, it is true.
        post_activation_bn (optional): bool
            Defines if you use batch normalization before or after the activation layer
            (there seems to be some advantages in some cases:
            https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
            If true, the batch normalization is used before the activation function,
            otherwise the activation comes first, as it is usually done. By default, it is false.
        activation_function (optional): string
            Keras activation function to be used. By default, 'relu'.

    References
        [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
            arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
        [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
            on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal', dropout_keep_prob=0.8,
                 kernel_size=17, preactivation=True, post_activation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.post_activation_bn = post_activation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, down_sample, n_filters_in):
        """Implement skip connection."""

        # Deal with down_sampling
        if down_sample > 1:
            y = MaxPooling1D(down_sample, strides=down_sample, padding='same')(y)
        elif down_sample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")

        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same', use_bias=False,
                       kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.post_activation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""

        x, y = inputs
        n_samples_in = y.shape[1]
        down_sample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, down_sample, n_filters_in)

        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)

        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=down_sample, padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)

        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)

            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)

            y = x
        return [x, y]


def get_model(n_classes, last_layer='sigmoid'):
    kernel_size = 16
    kernel_initializer = 'he_normal'
    signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
    x = signal
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x = Flatten()(x)
    diagn = Dense(n_classes, activation=last_layer,
                  kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn)
    return model


if __name__ == "__main__":
    model = get_model(6)
    model.summary()
