import tensorflow as tf
from tensorflow.keras import layers


class Envelope(layers.Layer):
    """
    This code is extracted from dimenet (https://github.com/gasteigerjo/dimenet).

    Envelope function that ensures a smooth cutoff.
    """
    def __init__(
        self,
        exponent,
        name='envelope'
    ):
        super().__init__(name=name)
        self.exponent = exponent

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def call(self, inputs):
        # Envelope function divided by r
        env_value = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)

        return tf.where(inputs < 1, env_value, tf.zeros_like(inputs))
