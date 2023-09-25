from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class ReadoutLayer(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_readout_fc_layers,
        name='readout_layer',
        num_targets=12,
        activation=None,
        kernel_initializer='glorot_uniform'
    ):
        super().__init__(name=name)

        self.fc_layers = list()
        for i in range(num_readout_fc_layers):
            self.fc_layers.extend([
                layers.Dense(hidden_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
                layers.BatchNormalization()])
        self.fc_layers.extend([
            layers.Dense(num_targets, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])
        self.fc_layers = Sequential(self.fc_layers)

    def call(self, final_embedding):
        return self.fc_layers(final_embedding)
