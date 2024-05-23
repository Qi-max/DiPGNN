import tensorflow as tf
from keras import backend as K
from dipgnn.utils.register import registers
from dipgnn.trainers.base_trainer import BaseTrainer


def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y,y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2

def pearson_r(y_true, y_pred):
    # A significant portion of the code is adapted from Keras_Metrics (https://github.com/WenYanger/Keras_Metrics).
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)


@registers.task.register("base_regression_trainer")
class BaseRegressionTrainer(BaseTrainer):
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        warmup_steps=None,
        decay_steps=100000,
        decay_rate=0.96,
        ema_decay=0.999,
        max_grad_norm=10.0,
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            ema_decay=ema_decay,
            max_grad_norm=max_grad_norm,
        )

        self.mae_func = tf.keras.metrics.MeanAbsoluteError()
        self.mse_func = tf.keras.metrics.MeanSquaredError()
        self.rmse_func = tf.keras.metrics.RootMeanSquaredError()

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            preds = self.model(inputs, validation_test=False, training=True)

            self.mae_func.update_state(targets, preds)
            self.mse_func.update_state(targets, preds)
            self.rmse_func.update_state(targets, preds)

            mae = self.mae_func.result()
            mse = self.mse_func.result()
            rmse = self.rmse_func.result()
            r2 = R_squared(targets, preds)
            pearson = pearson_r(targets, preds)

            self.mae_func.reset_states()
            self.mse_func.reset_states()
            self.rmse_func.reset_states()

            loss = tf.losses.mean_squared_error(targets, preds)

        self.update_weights(loss, tape)
        metrics.update_state(loss, mae, mse, rmse, r2, pearson, 1)
        return loss, targets, preds

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)

        preds = self.model(inputs, validation_test=True, training=False)

        self.mae_func.update_state(targets, preds)
        self.mse_func.update_state(targets, preds)
        self.rmse_func.update_state(targets, preds)

        mae = self.mae_func.result()
        mse = self.mse_func.result()
        rmse = self.rmse_func.result()
        r2 = R_squared(targets, preds)
        pearson = pearson_r(targets, preds)

        self.mae_func.reset_states()
        self.mse_func.reset_states()
        self.rmse_func.reset_states()

        loss = tf.losses.mean_squared_error(targets, preds)

        metrics.update_state(loss, mae, mse, rmse, r2, pearson, 1)
        return loss, targets, preds
