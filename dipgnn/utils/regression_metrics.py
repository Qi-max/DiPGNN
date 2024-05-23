import tensorflow as tf


class RegressionMetrics:
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """

    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex
        self.loss_metric = tf.keras.metrics.Mean()
        self.mean_mae_metric = tf.keras.metrics.Mean()
        self.mean_mse_metric = tf.keras.metrics.Mean()
        self.mean_rmse_metric = tf.keras.metrics.Mean()
        self.mean_r2_metric = tf.keras.metrics.Mean()
        self.mean_pearson_metric = tf.keras.metrics.Mean()

    def update_state(self, loss, mae, mse, rmse, r2, pearson, nsamples):
        self.loss_metric.update_state(loss, sample_weight=nsamples)
        self.mean_mae_metric.update_state(mae, sample_weight=nsamples)
        self.mean_mse_metric.update_state(mse, sample_weight=nsamples)
        self.mean_rmse_metric.update_state(rmse, sample_weight=nsamples)
        self.mean_r2_metric.update_state(r2, sample_weight=nsamples)
        self.mean_pearson_metric.update_state(pearson, sample_weight=nsamples)

    def write(self):
        for key, value in self.result().items():
            tf.summary.scalar(key, value)
            if self.ex is not None:
                if key not in self.ex.current_run.info:
                    self.ex.current_run.info[key] = []
                self.ex.current_run.info[key].append(value)

        if self.ex is not None:
            if f'step_{self.tag}' not in self.ex.current_run.info:
                self.ex.current_run.info[f'step_{self.tag}'] = []
            self.ex.current_run.info[f'step_{self.tag}'].append(tf.summary.experimental.get_step())

    def reset_states(self):
        self.loss_metric.reset_states()
        self.mean_mae_metric.reset_states()
        self.mean_mse_metric.reset_states()
        self.mean_rmse_metric.reset_states()
        self.mean_r2_metric.reset_states()
        self.mean_pearson_metric.reset_states()

    def keys(self):
        keys = [f'loss_{self.tag}', f'mean_mae_{self.tag}', f'mean_mse_{self.tag}',
                f'mean_rmse_{self.tag}', f'mean_r2_{self.tag}', f'mean_pearson_{self.tag}']
        keys.extend(["{}_{}".format(key, self.tag) for key in self.targets])
        return keys

    def result(self):
        result_dict = {}
        result_dict[f'loss_{self.tag}'] = self.loss
        result_dict[f'mean_mae_{self.tag}'] = self.mean_mae
        result_dict[f'mean_mse_{self.tag}'] = self.mean_mse
        result_dict[f'mean_rmse_{self.tag}'] = self.mean_rmse
        result_dict[f'mean_r2_{self.tag}'] = self.mean_r2
        result_dict[f'mean_pearson_{self.tag}'] = self.mean_pearson
        return result_dict

    @property
    def loss(self):
        return self.loss_metric.result().numpy().item()

    @property
    def mean_mae(self):
        return self.mean_mae_metric.result().numpy().item()

    @property
    def mean_mse(self):
        return self.mean_mse_metric.result().numpy().item()

    @property
    def mean_rmse(self):
        return self.mean_rmse_metric.result().numpy().item()

    @property
    def mean_r2(self):
        return self.mean_r2_metric.result().numpy().item()

    @property
    def mean_pearson(self):
        return self.mean_pearson_metric.result().numpy().item()
