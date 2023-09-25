from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ResidualLayer(layers.Layer):
    def __init__(
        self,
        tensor_size,
        name='residual_layer',
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    ):
        super().__init__(name=name)
        self.fc_layers = Sequential([
            layers.Dense(
                tensor_size, activation=activation, use_bias=use_bias,
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
            layers.BatchNormalization(),
            layers.Dense(
                tensor_size, activation=activation, use_bias=use_bias,
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
            layers.BatchNormalization()])

    def call(self, tensors):
        output = tensors + self.fc_layers(tensors)
        return output
