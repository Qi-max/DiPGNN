import tensorflow as tf


class ClassificationMetrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex
        self.loss_metric = tf.keras.metrics.Mean()
        self.mean_auc_metric = tf.keras.metrics.Mean()

    def update_state(self, loss, auc, nsamples):
        self.loss_metric.update_state(loss, sample_weight=nsamples)
        self.mean_auc_metric.update_state(auc, sample_weight=nsamples)

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
        self.mean_auc_metric.reset_states()

    def keys(self):
        keys = [f'loss_{self.tag}', f'mean_auc_{self.tag}']
        keys.extend([key + '_' + self.tag for key in self.targets])
        return keys

    def result(self):
        result_dict = {
            f'loss_{self.tag}': self.loss,
            f'mean_auc_{self.tag}': self.mean_auc
        }
        return result_dict

    @property
    def loss(self):
        return self.loss_metric.result().numpy().item()

    @property
    def mean_auc(self):
        return self.mean_auc_metric.result().numpy().item()
