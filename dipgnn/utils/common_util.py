import argparse
import logging
import importlib
from pathlib import Path
import tensorflow as tf
from dipgnn.utils.register import registers


class CommonArgs():
    def __init__(self):
        self.parser = argparse.ArgumentParser('DiPGNN')
        self.add_common_args()

    def get_args(self):
        return self.args

    def add_common_args(self):
        add_arg = self.parser.add_argument

        add_arg('--graph_data_file_df', type=str, default=None, help='graph_data_path_df')
        add_arg('--targets_data_file', type=str, default=None, help='targets_data_file')
        add_arg('--output_path', type=str, default="./classification/results", help='output_path')

        add_arg('--trainer_name', type=str, default="base_classification_trainer", help='trainer_name')
        add_arg('--task_name', type=str, default="classification_training_task", help='task_name')
        add_arg('--model_name', type=str, default="dipgnn", help='model_name')
        add_arg('--data_container_name', type=str, default="data_container", help='data_container_name')
        add_arg('--data_provider_name', type=str, default="data_provider", help='data_provider_name')

        add_arg('--random_seed', type=int, default=1992, help='random_seed')
        add_arg('--shuffle', action='store_true', help='shuffle')

        add_arg('--batch_size', default=None, type=int, help='batch_size')
        add_arg('--train', default=0.8, type=str, help='float, int or list for training set')
        add_arg('--validation', default=0.1, type=str, help='float, int or list for validation set')
        add_arg('--test', default=0.1, type=str, help='float, int or list for test set')
        add_arg('--cutoff', type=float, default=4.5, help='cutoff')

        add_arg('--task_type', default="classification", type=str, help='regression or classification')
        add_arg('--use_sigmoid', action='store_true', help='use sigmoid or softmax')

        add_arg('--target_type', default="atom", type=str, help='structure, atom or path')
        add_arg('--target_col', type=str, default="targets", help='target_col')
        add_arg('--target_name', type=str, default="", help='target_name')
        add_arg('--num_targets', type=int, default=2, help='num_targets')

        add_arg('--num_layers', type=int, default=3, help='num_layers')

        add_arg('--input_size', type=int, default=2, help='input_size')
        add_arg('--atom_embedding_size', type=int, default=32, help='atom_embedding_size')
        add_arg('--bond_embedding_size', type=int, default=32, help='bond_embedding_size')
        add_arg('--hidden_size', type=int, default=32, help='hidden_size')

        add_arg('--rbf', type=str, default="Bessel", help='Bessel or Gaussian')
        add_arg('--sbf', type=str, default="Spherical", help='Spherical or Gaussian')
        add_arg('--num_spherical', type=int, default=7, help='num_spherical')
        add_arg('--num_radial', type=int, default=6, help='num_radial')
        add_arg('--num_gaussian', type=int, default=50, help='num_gaussian')
        add_arg('--gaussian_radial_var', type=float, default=0.2, help='gaussian_radial_var')
        add_arg('--gaussian_angular_var', type=float, default=0.2, help='gaussian_angular_var')
        add_arg('--envelope_exponent', type=int, default=5, help='envelope_exponent')

        add_arg('--num_embedding_fc_layers', type=int, default=2, help='num_embedding_fc_layers')
        add_arg('--num_b2b_res_layers', type=int, default=2, help='num_b2b_res_layers')
        add_arg('--num_readout_fc_layers', type=int, default=3, help='num_readout_fc_layers')
        add_arg('--feature_add_or_concat', type=str, default="add", help='feature_add_or_concat')

        add_arg('--embedding_dropout', type=float, default=0.7, help='embedding_dropout')
        add_arg('--output_dropout', type=float, default=0.7, help='output_dropout')

        add_arg('--kernel_initializer', type=str, default="GlorotOrthogonal", help='kernel_initializer')
        add_arg('--num_steps', type=int, default=3000000, help='num_steps')
        add_arg('--early_stopping_epochs', type=int, default=500, help="stop training if score does not be better for these epochs")
        add_arg('--ema_decay', type=float, default=0.999, help='ema_decay')
        add_arg('--max_grad_norm', type=float, default=10.0, help='max_grad_norm')

        add_arg('--learning_rate', type=float, default=0.0002, help='learning_rate')
        add_arg('--l2_loss_decay', type=float, default=0.000, help='l2_loss_decay')
        add_arg('--use_huber_loss', action='store_true', help='use_huber_loss')
        add_arg('--huber_delta', type=float, default=1.0, help='huber_delta')
        add_arg('--warmup_steps', type=int, default=3000, help='warmup_steps')
        add_arg('--decay_rate', type=float, default=0.1, help='decay_rate')
        add_arg('--decay_steps', type=int, default=4000000, help='decay_steps')

        add_arg("--save_all_models", action='store_true', help='save_all_models')
        add_arg("--save_all_predictions", action='store_true', help='save_all_predictions')
        add_arg("--ckpt_max_to_keep", type=int, default=3, help='ckpt_max_to_keep')
        add_arg('--ckpt_path', type=str, default=None, help='checkpoint path, default is None')
        add_arg('--comment', type=str, default='', help='comment')

        self.args = self.parser.parse_args()
        logging.info("Common args is: {}".format(self.args))

def _import_local_file(path: Path, *, project_root: Path) -> None:
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    :type project_root: Path
    """

    path = path.resolve()
    project_root = project_root.resolve()

    module_name = ".".join(
        path.absolute()
        .relative_to(project_root.absolute())
        .with_suffix("")
        .parts
    )
    logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L134.
def setup_imports():
    # First, check if imports are already setup
    if registers.already_setup:
        return

    try:
        project_root = Path(__file__).resolve().absolute().parent.parent.parent
        logging.info(f"Project root: {project_root}")

        import_keys = ["data", "models", "tasks", "trainers"]
        for key in import_keys:
            for f in (project_root / "dipgnn" / key).rglob("*.py"):
                _import_local_file(f, project_root=project_root)
    finally:
        registers.already_setup = True


class LinearWarmupExponentialDecay(tf.optimizers.schedules.LearningRateSchedule):
    """
    This code is extracted from dimenet (https://github.com/gasteigerjo/dimenet).
    
    This schedule combines a linear warmup with an exponential decay.
    """
    def __init__(self, learning_rate, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.warmup = tf.optimizers.schedules.PolynomialDecay(
            1 / warmup_steps, warmup_steps, end_learning_rate=1)
        self.decay = tf.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps, decay_rate)
        self.learning_rate = learning_rate

    def __call__(self, step):
        self.learning_rate = self.warmup(step) * self.decay(step)
        return self.learning_rate

    def get_learning_rate(self):
        return self.learning_rate
