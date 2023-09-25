import tensorflow as tf
from tensorflow.keras import layers


class GaussianExpansion(layers.Layer):
    def __init__(
        self,
        dmin,
        dmax,
        num_gaussian=50,
        var=None
    ):
        super().__init__(name="gaussian")
        self.filter = tf.linspace(dmin, dmax, num_gaussian)
        self.var = (dmax - dmin)/num_gaussian if var is None else var

    def call(self, tensors):
        return tf.exp(-(tensors[..., tf.newaxis] - self.filter)**2 / self.var**2)
