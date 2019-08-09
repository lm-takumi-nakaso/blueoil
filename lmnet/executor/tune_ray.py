# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import six
import click
import tensorflow as tf
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.keras.utils import Progbar
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from functools import partial
import sys
import traceback

from easydict import EasyDict
from lmnet.utils import executor, config as config_util
from lmnet.datasets.dataset_iterator import DatasetIterator

import ray
from ray.tune import run_experiments, Trainable, Experiment
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.logger import DEFAULT_LOGGERS, TFLogger
from ray.tune.result import EPISODE_REWARD_MEAN, TRAINING_ITERATION

if six.PY2:
    import subprocess32 as subprocess
else:
    import subprocess


def subproc_call(cmd, timeout=None):
    """
    Execute a command with timeout, and return both STDOUT/STDERR.
    Args:
        cmd(str): the command to execute.
        timeout(float): timeout in seconds.
    Returns:
        output(bytes), retcode(int). If timeout, retcode is -1.
    """
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT,
            shell=True, timeout=timeout)
        return output, 0
    except subprocess.TimeoutExpired as e:
        print("Command '{}' timeout!".format(cmd))
        print(e.output.decode('utf-8'))
        return e.output, -1
    except subprocess.CalledProcessError as e:
        print("Command '{}' failed, return code={}".format(cmd, e.returncode))
        print(e.output.decode('utf-8'))
        return e.output, e.returncode
    except Exception:
        print("Command '{}' failed to run.".format(cmd))
        return "", -2


def get_num_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """

    def warn_return(ret, message):
        built_with_cuda = tf.test.is_built_with_cuda()
        if not built_with_cuda and ret > 0:
            print(message + "But TensorFlow was not built with CUDA support and could not use GPUs!")
        return ret

    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env:
        return warn_return(len(env.split(',')), "Found non-empty CUDA_VISIBLE_DEVICES. ")
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code == 0:
        output = output.decode('utf-8')
        return warn_return(len(output.strip().split('\n')), "Found nvidia-smi. ")
    else:
        print('Not working for this one... But there are other methods you can try...')
        raise NotImplementedError


def get_best_trial(trial_list, metric):
    """Retrieve the best trial."""
    return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))


def trial_str_creator(trial):
    """Rename trial to shorter string"""
    return "{}_{}".format(trial.trainable_name, trial.trial_id)


def get_best_result(trial_list, metric, param):
    """Retrieve the last result from the best trial."""
    return {metric: get_best_trial(trial_list, metric).last_result[metric],
            param: get_best_trial(trial_list, metric).last_result[param]}


def save_checkpoint(saver, sess, global_step, step, environment):
    checkpoint_file = "save.ckpt"
    saver.save(
        sess,
        os.path.join(environment.CHECKPOINT_DIR, checkpoint_file),
        global_step=global_step,
    )

    if step == 0:
        # check create pb on only first step.
        minimal_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            ["output"],
        )
        pb_name = "minimal_graph_with_shape_{}.pb".format(step + 1)
        pbtxt_name = "minimal_graph_with_shape_{}.pbtxt".format(step + 1)
        tf.train.write_graph(minimal_graph, environment.CHECKPOINT_DIR, pb_name, as_text=False)
        tf.train.write_graph(minimal_graph, environment.CHECKPOINT_DIR, pbtxt_name, as_text=True)


def setup_dataset(config, subset, rank, processes):
    DatasetClass = config.DATASET_CLASS
    dataset_kwargs = dict((key.lower(), val) for key, val in config.DATASET.items())
    dataset = DatasetClass(subset=subset, **dataset_kwargs)
    processes = dataset_kwargs.pop("enable_prefetch", False) and processes
    return DatasetIterator(dataset, seed=rank, enable_prefetch=processes)


def start_training(config, environment, num_cpus, recreate=False, seed=0):
    config_util.display(config)
    executor.init_logging(config)

    executor.prepare_dirs(recreate=recreate)
    config_util.save_yaml(environment.EXPERIMENT_DIR, config)

    ModelClass = config.NETWORK_CLASS
    network_kwargs = dict((key.lower(), val) for key, val in config.NETWORK.items())

    train_dataset = setup_dataset(config, "train", seed, processes=num_cpus-1)
    print("train dataset num:", train_dataset.num_per_epoch)

    graph = tf.Graph()
    with graph.as_default():
        if ModelClass.__module__.startswith("lmnet.networks.object_detection"):
            model = ModelClass(
                classes=train_dataset.classes,
                num_max_boxes=train_dataset.num_max_boxes,
                is_debug=config.IS_DEBUG,
                **network_kwargs,
            )
        else:
            model = ModelClass(
                classes=train_dataset.classes,
                is_debug=config.IS_DEBUG,
                **network_kwargs,
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        is_training_placeholder = tf.placeholder(tf.bool, name="is_training_placeholder")

        images_placeholder, labels_placeholder = model.placeholderes()

        output = model.inference(images_placeholder, is_training_placeholder)
        if ModelClass.__module__.startswith("lmnet.networks.object_detection"):
            loss = model.loss(output, labels_placeholder, global_step)
        else:
            loss = model.loss(output, labels_placeholder)
        opt = model.optimizer(global_step)
        train_op = model.train(loss, opt, global_step)
        metrics_ops_dict, metrics_update_op = model.metrics(output, labels_placeholder)
        # TODO(wakisaka): Deal with many networks.
        model.summary(output, labels_placeholder)

        summary_op = tf.summary.merge_all()

        metrics_summary_op, metrics_placeholders = executor.prepare_metrics(metrics_ops_dict)

        init_op = tf.global_variables_initializer()
        reset_metrics_op = tf.local_variables_initializer()

        saver = tf.train.Saver(max_to_keep=None)

        if config.IS_PRETRAIN:
            all_vars = tf.global_variables()
            pretrain_var_list = [
                var for var in all_vars if var.name.startswith(tuple(config.PRETRAIN_VARS))
            ]
            print("pretrain_vars", [
                var.name for var in pretrain_var_list
            ])
            pretrain_saver = tf.train.Saver(pretrain_var_list, name="pretrain_saver")

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))  # tf.ConfigProto(log_device_placement=True)
    # TODO(wakisaka): XLA JIT
    # session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    sess = tf.Session(graph=graph, config=session_config)
    sess.run([init_op, reset_metrics_op])

    train_writer = tf.summary.FileWriter(environment.TENSORBOARD_DIR + "/train", sess.graph)

    if config.IS_PRETRAIN:
        print("------- Load pretrain data ----------")
        pretrain_saver.restore(sess, os.path.join(config.PRETRAIN_DIR, config.PRETRAIN_FILE))
        sess.run(tf.assign(global_step, 0))

    last_step = sess.run(global_step)

    # Calculate max steps. The priority of config.MAX_EPOCHS is higher than config.MAX_STEPS.
    if "MAX_EPOCHS" in config:
        max_steps = int(train_dataset.num_per_epoch / config.BATCH_SIZE * config.MAX_EPOCHS)
    else:
        max_steps = config.MAX_STEPS

    progbar = Progbar(max_steps)
    progbar.update(last_step)
    for step in range(last_step, max_steps):
        images, labels = train_dataset.feed()

        feed_dict = {
            is_training_placeholder: True,
            images_placeholder: images,
            labels_placeholder: labels,
        }
        sess.run([train_op], feed_dict=feed_dict)

        to_be_saved = step == 0 or (step + 1) == max_steps or (step + 1) % config.SAVE_STEPS == 0

        if step * ((step + 1) % config.SUMMARISE_STEPS) == 0:
            # Runtime statistics for develop.
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            sess.run(reset_metrics_op)
            summary, _ = sess.run(
                [summary_op, metrics_update_op], feed_dict=feed_dict,
                # options=run_options,
                # run_metadata=run_metadata,
            )
            # train_writer.add_run_metadata(run_metadata, "step: {}".format(step + 1))
            train_writer.add_summary(summary, step + 1)

            metrics_values = sess.run(list(metrics_ops_dict.values()))
            metrics_feed_dict = {placeholder: value for placeholder, value in zip(metrics_placeholders, metrics_values)}

            metrics_summary, = sess.run(
                [metrics_summary_op], feed_dict=metrics_feed_dict,
            )
            train_writer.add_summary(metrics_summary, step + 1)

            train_writer.flush()

            metrics_dict = sess.run(metrics_ops_dict)
            if config.NETWORK_CLASS.__module__.startswith("lmnet.networks.segmentation"):
                episode_reward_mean = metrics_dict['mean_iou']
            elif config.NETWORK_CLASS.__module__.startswith("lmnet.networks.object_detection"):
                episode_reward_mean = metrics_dict['MeanAveragePrecision_0.5']
            else:
                episode_reward_mean = metrics_dict['accuracy']
            yield {**{TRAINING_ITERATION: step + 1}, **{EPISODE_REWARD_MEAN: episode_reward_mean}, **metrics_dict}

        progbar.update(step + 1)
    # training loop end.
    print("Done")


class Trainer(Trainable):
    """ TrainTunable class interfaces with Ray framework """

    def _setup(self, config):

        self.__log = open('trial.log', mode='w', buffering=1)
        self.__start_training = None

    def _train(self):
        if self.__start_training is None:
            from lmnet import environment

            environment.EXPERIMENT_DIR = os.path.abspath('.')
            environment.TENSORBOARD_DIR = os.path.abspath('tensorboard')
            environment.CHECKPOINTS_DIR = os.path.abspath('checkpoints')
            environment._init_flag = True

            chosen_kwargs =self.config
            config_util.save_yaml(os.path.join(environment.EXPERIMENT_DIR, 'trial'), chosen_kwargs)
            kwargs = config_util.load(chosen_kwargs['lm_config'])
            config = EasyDict(kwargs['UPDATE_PARAMETERS_FOR_EACH_TRIAL'](kwargs, chosen_kwargs))
            if config is None:
                config = kwargs
            config = EasyDict(config)

            num_cpus = len(ray.get_resource_ids()['CPU'])

            self.__start_training = start_training(config=config, environment=environment, num_cpus=num_cpus)

        with redirect_stdout(self.__log), redirect_stderr(self.__log):
            try:
                result = next(self.__start_training)
            except:
                traceback.print_exc()
                raise

        return result

    def _stop(self):
        os._exit(0)


def run_train(config_file, redis_address, num_cpus, num_gpus, log_to_driver, temp_dir, max_concurrent, num_samples):
    from lmnet.environment import OUTPUT_DIR

    def easydict_to_dict(config):
        if isinstance(config, EasyDict):
            config = dict(config)

        for key, value in config.items():
            if isinstance(value, EasyDict):
                value = dict(value)
                easydict_to_dict(value)
            config[key] = value
        return config

    if num_cpus is None:
        num_cpus = len(os.sched_getaffinity(0))
    if num_gpus is None:
        num_gpus = get_num_gpu()
    if max_concurrent is None:
        max_concurrent = num_gpus

    # Expecting use of gpus to do parameter search
    if redis_address is not None:
        ray.init(redis_address=redis_address, log_to_driver=log_to_driver)
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, temp_dir=temp_dir, log_to_driver=log_to_driver)

    config_file = os.path.abspath(config_file)
    config_name, _ = os.path.splitext(os.path.basename(config_file))
    experiment_name = '{}_{:%Y%m%d%H%M%S}'.format(config_name, datetime.now())
    config = config_util.load(config_file)

    max_steps = config.get('MAX_STEPS', sys.maxsize)
    save_steps = config.get('SAVE_STEPS', 0)
    tune_space = config.TUNE_SPACE

    algo = HyperOptSearch(tune_space, max_concurrent=max_concurrent, metric=EPISODE_REWARD_MEAN, mode='max')

    scheduler = AsyncHyperBandScheduler(time_attr=TRAINING_ITERATION, metric=EPISODE_REWARD_MEAN, mode='max', max_t=max_steps)

    tune_spec = {
        'checkpoint_freq': save_steps,
        'config': {
            'lm_config': config_file
        },
        'local_dir': OUTPUT_DIR,
        'loggers': [logger for logger in DEFAULT_LOGGERS if logger != TFLogger],
        'name': experiment_name,
        'num_samples': num_samples,
        'resources_per_trial': {
            'cpu': num_cpus // num_gpus,
            'gpu': 1
        },
        'run': Trainer,
        'stop': {
            TRAINING_ITERATION: max_steps
        },
        'trial_name_creator': ray.tune.function(trial_str_creator),
    }

    trials = ray.tune.run_experiments([Experiment(**tune_spec)], search_alg=algo, scheduler=scheduler)
    print("The best result is", get_best_result(trials, metric=EPISODE_REWARD_MEAN, param='config'))


@click.command('train', context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '-c',
    '--config_file',
    '--config-file',
    help="config file path for this training",
    required=True,
)
@click.option(
    '--redis_address',
    '--redis-address',
    help='[optional] the address to use for connecting to Redis',
    default=None,
)
@click.option(
    '--num_cpus',
    '--num-cpus',
    help='[optional] the number of CPUs on this node',
    default=None,
    type=int
)
@click.option(
    '--num_gpus',
    '--num-gpus',
    help='[optional] the number of GPUs on this node',
    default=None,
    type=int
)
@click.option(
    '--log_to_driver',
    '--log-to-driver',
    help='[optional] If true, then output from all of the worker processes on all nodes will be directed to the driver',
    default=True,
    type=int
)
@click.option(
    '--temp_dir',
    '--temp-dir',
    help='[optional] If provided, it will specify the root temporary directory for the Ray process',
    default=False,
)
@click.option(
    '--max_concurrent',
    '--max-concurrent',
    help='[optional] Number of maximum concurrent trials',
    default=None,
    type=int
)
@click.option(
    '--num_samples',
    '--num-samples',
    help='[optional] Number of times to sample from the hyperparameter space',
    default=sys.maxsize,
    type=int
)
def main(config_file, redis_address, num_cpus, num_gpus, log_to_driver, temp_dir, max_concurrent, num_samples):
    run_train(config_file, redis_address, num_cpus, num_gpus, log_to_driver, temp_dir, max_concurrent, num_samples)


if __name__ == '__main__':
    main()
