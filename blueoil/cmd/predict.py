# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
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
import imghdr
import math
import os
from glob import glob

import numpy as np
import tensorflow as tf

from blueoil import environment
from blueoil.utils.image import load_image
from blueoil.utils import config as config_util
from blueoil.utils.executor import search_restore_filename
from blueoil.utils.predict_output.writer import OutputWriter

DUMMY_FILENAME = "DUMMY_FILE"


def _get_images(start_index, image_inputs, pre_processor, data_format):
    images = []
    raw_images = []
    image_keys = []

    for image_index, image_input in enumerate(image_inputs, start_index):
        if image_input == DUMMY_FILENAME:
            raw_image = np.zeros_like(raw_images[-1])
        else:
            try:
                image_file = os.fspath(image_input)
            except TypeError:
                image_file = None
            if image_file is not None:
                image_key = image_file
                raw_image = load_image(image_file)
            else:
                image_key = image_index
                raw_image = image_input
        image = pre_processor(image=raw_image)['image']
        if data_format == 'NCHW':
            image = np.transpose(image, [2, 0, 1])

        images.append(image)
        raw_images.append(raw_image)
        image_keys.append(image_key)

    return np.array(images), np.array(raw_images), image_keys


def _all_image_files(directory):
    return sorted((
        os.path.abspath(file_path)
        for file_path in glob(os.path.join(directory, "*"))
        if os.path.isfile(file_path) and imghdr.what(file_path) in {"jpeg", "png"}
    ))


def _run(inputs, output_dir, config, restore_path, save_images):
    ModelClass = config.NETWORK_CLASS
    network_kwargs = {key.lower(): val for key, val in config.NETWORK.items()}

    graph = tf.Graph()
    with graph.as_default():
        model = ModelClass(
            classes=config.CLASSES,
            is_debug=config.IS_DEBUG,
            **network_kwargs
        )

        is_training = tf.constant(False, name="is_training")

        images_placeholder, _ = model.placeholders()
        output_op = model.inference(images_placeholder, is_training)

        init_op = tf.compat.v1.global_variables_initializer()

        saver = tf.compat.v1.train.Saver(max_to_keep=None)

    session_config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    sess.run(init_op)
    saver.restore(sess, restore_path)

    try:
        input_dir = os.fspath(inputs)
    except TypeError:
        input_dir = None
    if input_dir is not None:
        all_image_inputs = _all_image_files(input_dir)
    else:
        all_image_inputs = list(inputs)

    step_size = int(math.ceil(len(all_image_inputs) / config.BATCH_SIZE))

    writer = OutputWriter(
        task=config.TASK,
        classes=config.CLASSES,
        image_size=config.IMAGE_SIZE,
        data_format=config.DATA_FORMAT
    )

    results = []
    for step in range(step_size):
        start_index = step * config.BATCH_SIZE
        end_index = (step + 1) * config.BATCH_SIZE

        image_inputs = all_image_inputs[start_index:end_index]

        if len(image_inputs) < config.BATCH_SIZE:
            # add dummy image.
            image_inputs += [DUMMY_FILENAME] * (config.BATCH_SIZE - len(image_inputs))

        images, raw_images, image_keys = _get_images(
            start_index, image_inputs, config.DATASET.PRE_PROCESSOR, config.DATA_FORMAT)

        outputs = sess.run(output_op, feed_dict={images_placeholder: images})

        if config.POST_PROCESSOR:
            outputs = config.POST_PROCESSOR(outputs=outputs)["outputs"]

        results.append(outputs)

        writer.write(output_dir, outputs, raw_images, image_keys, step, save_material=save_images)

    sess.close()

    return results


def run(inputs, output_dir, experiment_id, config_file, restore_path, save_images):
    environment.init(experiment_id)
    config = config_util.load_from_experiment()
    if config_file:
        config = config_util.merge(config, config_util.load(config_file))

    if restore_path is None:
        restore_file = search_restore_filename(environment.CHECKPOINTS_DIR)
        restore_path = os.path.join(environment.CHECKPOINTS_DIR, restore_file)

    print("Restore from {}".format(restore_path))

    if not os.path.exists("{}.index".format(restore_path)):
        raise FileNotFoundError("Checkpoint file not found: '{}'".format(restore_path))

    print("---- start predict ----")

    return _run(inputs, output_dir, config, restore_path, save_images)

    print("---- end predict ----")


def predict(input_dir, output_dir, experiment_id, config_file=None, checkpoint=None, save_images=True):
    """Make predictions from input dir images by using trained model.
        Save the predictions npy, json, images results to output dir.
        npy: `{output_dir}/npy/{batch number}.npy`
        json: `{output_dir}/json/{batch number}.json`
        images: `{output_dir}/images/{some type}/{input image file name}`
    """
    restore_path = None
    if checkpoint:
        saved_dir = os.environ.get("OUTPUT_DIR", "saved")
        restore_path = os.path.join(saved_dir, experiment_id, "checkpoints", checkpoint)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError("Input directory not found: '{}'".format(input_dir))

    return run(input_dir, output_dir, experiment_id, config_file, restore_path, save_images)
