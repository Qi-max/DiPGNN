import tensorflow as tf


class ClassificationMetrics:
    def __init__(self, tag, targets, ex=None):
        self.tag = tag
        self.targets = targets
        self.ex = ex
        self.loss_metric = tf.keras.metrics.Mean()
        self.mean_acc_metric = tf.keras.metrics.Mean()
        self.mean_recall_metric = tf.keras.metrics.Mean()
        self.mean_precision_metric = tf.keras.metrics.Mean()
        self.mean_auc_metric = tf.keras.metrics.Mean()
        self.mean_f1_metric = tf.keras.metrics.Mean()

    def update_state(self, loss, acc, recall, precision, auc, f1, nsamples):
        self.loss_metric.update_state(loss, sample_weight=nsamples)
        self.mean_acc_metric.update_state(acc, sample_weight=nsamples)
        self.mean_recall_metric.update_state(recall, sample_weight=nsamples)
        self.mean_precision_metric.update_state(precision, sample_weight=nsamples)
        self.mean_auc_metric.update_state(auc, sample_weight=nsamples)
        self.mean_f1_metric.update_state(f1, sample_weight=nsamples)

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
        self.mean_acc_metric.reset_states()
        self.mean_recall_metric.reset_states()
        self.mean_precision_metric.reset_states()
        self.mean_auc_metric.reset_states()
        self.mean_f1_metric.reset_states()

    def keys(self):
        keys = [f'loss_{self.tag}', f'mean_acc_{self.tag}', f'mean_recall_{self.tag}',
                f'mean_precision_{self.tag}', f'mean_auc_{self.tag}', f'mean_f1_{self.tag}']
        keys.extend([key + '_' + self.tag for key in self.targets])
        return keys

    def result(self):
        result_dict = {
            f'loss_{self.tag}': self.loss,
            f'mean_acc_{self.tag}': self.mean_acc,
            f'mean_recall_{self.tag}': self.mean_recall,
            f'mean_precision_{self.tag}': self.mean_precision,
            f'mean_auc_{self.tag}': self.mean_auc,
            f'mean_f1_{self.tag}': self.mean_f1
        }
        return result_dict

    @property
    def loss(self):
        return self.loss_metric.result().numpy().item()

    @property
    def mean_acc(self):
        return self.mean_acc_metric.result().numpy().item()

    @property
    def mean_recall(self):
        return self.mean_recall_metric.result().numpy().item()

    @property
    def mean_precision(self):
        return self.mean_precision_metric.result().numpy().item()

    @property
    def mean_auc(self):
        return self.mean_auc_metric.result().numpy().item()

    @property
    def mean_f1(self):
        return self.mean_f1_metric.result().numpy().item()

