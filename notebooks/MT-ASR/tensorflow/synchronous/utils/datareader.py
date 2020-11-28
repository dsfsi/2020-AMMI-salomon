from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
from utils import tfAudioTools as tfAudio
from utils.tfRecord import RESERVED_TOKENS_TO_INDEX

DELAY_SYMBOL = RESERVED_TOKENS_TO_INDEX["<DELAY>"]
L2_SYMBOL = RESERVED_TOKENS_TO_INDEX["<2L2>"]

def get_input_fn(mode,
                 hparams,
                 transform=True):
    """Provides input to the graph, either from disk or via a placeholder.
    """

    def input_fn():
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            prefix = os.path.join(hparams.data_dir, "train.*.record")
        else:
#             prefix = os.path.join(hparams.data_dir, "test2015.0.record")
            prefix = os.path.join(hparams.data_dir, "test.0.record")
            
        data_file_patterns = sorted(glob.glob(prefix))
        drop_long_sequences = mode == tf.contrib.learn.ModeKeys.TRAIN

        with tf.name_scope("input_queues"):
            with tf.device("/cpu:0"):
                if mode == tf.contrib.learn.ModeKeys.TRAIN:
                    filename_queue = tf.train.string_input_producer(
                        data_file_patterns, num_epochs=None, shuffle=True)
                else:
                    filename_queue = tf.train.string_input_producer(
                        data_file_patterns, num_epochs=1, shuffle=False)

                reader_tfRecord = tf.TFRecordReader()
                _, serialized_example = reader_tfRecord.read(filename_queue)
                features = tf.parse_single_example(
                    serialized_example,
                    features={'inputs': tf.FixedLenFeature([], tf.string),
                              'target_l1': tf.FixedLenFeature([], tf.string),
                              'target_l2': tf.FixedLenFeature([], tf.string)}
                )

                inputs = tf.reshape(tf.decode_raw(features['inputs'], tf.float32),
                                    [-1, hparams.dim_feature])
                                    
                # TODO automatically change 3000
                inputs = inputs[:3000, :] if drop_long_sequences else inputs

                if transform:
                    inputs = process_raw_feature(inputs, hparams.dim_feature, hparams.num_context, hparams.downsample)
                target_l1 = tf.decode_raw(features['target_l1'], tf.int32)
                target_l2 = tf.decode_raw(features['target_l2'], tf.int32)
                
                # Manually Add delay (for wait-k)
                delay_target_l2 = tf.concat([tf.constant([DELAY_SYMBOL, DELAY_SYMBOL, DELAY_SYMBOL, L2_SYMBOL], tf.int32),
                                             target_l2[1:]], 0)

                feature_map = {"inputs": inputs, "targets_l1": target_l1,
                               "targets_l2": delay_target_l2}

                if mode == tf.contrib.learn.ModeKeys.TRAIN:
                    feature_map = fentch_batch_bucket(feature_map)
                else:
                    feature_map = fentch_batch(feature_map)

                targets_l1 = feature_map["targets_l1"]
                targets_l2 = feature_map["targets_l2"]

                targets_l1_length = tf.shape(targets_l1)[1]
                targets_l2_length = tf.shape(targets_l2)[1]

                targets_l1_pad, targets_l2_pad = tf.cond(
                    tf.less(targets_l1_length, targets_l2_length),
                    lambda: (tf.pad(targets_l1, [[0, 0], [0, targets_l2_length - targets_l1_length]]), targets_l2),
                    lambda: (targets_l1, tf.pad(targets_l2, [[0, 0], [0, targets_l1_length - targets_l2_length]])))
                    
                feature_map["targets_l1"] = targets_l1_pad
                feature_map["targets_l2"] = targets_l2_pad

            # Ensure inputs and targets are proper rank.
            while len(feature_map["inputs"].get_shape()) != 3:
                feature_map["inputs"] = tf.expand_dims(feature_map["inputs"], axis=-1)
            while len(feature_map["targets_l1"].get_shape()) != 4:
                feature_map["targets_l1"] = tf.expand_dims(feature_map["targets_l1"], axis=-1)
            while len(feature_map["targets_l2"].get_shape()) != 4:
                feature_map["targets_l2"] = tf.expand_dims(feature_map["targets_l2"], axis=-1)

        rand_inputs, rand_target_l1, rand_target_l2 = \
            feature_map["inputs"], feature_map["targets_l1"], feature_map["targets_l2"]

        # Set shapes so the ranks are clear.
        rand_inputs.set_shape([None, None, None])
        rand_target_l1.set_shape([None, None, None, None])
        rand_target_l2.set_shape([None, None, None, None])

        # Final feature map.
        rand_feature_map = {"inputs": rand_inputs, "targets_l2": rand_target_l2}
        return rand_feature_map, rand_target_l1

    return input_fn


def fentch_batch(features):
    list_inputs = [features["inputs"], features["targets_l1"], features["targets_l2"]]
    list_outputs = tf.train.batch(
        tensors=list_inputs,
        batch_size=8,
        num_threads=1,
        capacity=2000,
        dynamic_pad=True,
        allow_smaller_final_batch=True
    )
    feature_map = {"inputs": list_outputs[0], "targets_l1": list_outputs[1],
                   "targets_l2": list_outputs[2]}
    return feature_map


def fentch_batch_bucket(features):
    """
    the input tensor length is not equal,
    so will add the len as a input tensor
    list_inputs: [tensor1, tensor2]
    added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
    """
    batch_size_list = [80, 64, 48, 32, 24, 16, 12, 8, 4]
    bucket_boundaries_list = [100, 200, 404, 615, 828, 1065, 1360, 1792]
    list_inputs = [features["inputs"], features["targets_l1"], features["targets_l2"]]
    _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.shape(features["inputs"])[0],
        tensors=list_inputs,
        batch_size=batch_size_list,
        bucket_boundaries=bucket_boundaries_list,
        num_threads=8,
        bucket_capacities=[i * 3 for i in batch_size_list],
        capacity=2000,
        dynamic_pad=True,
        allow_smaller_final_batch=True)
    feature_map = {"inputs": list_outputs[0], "targets_l1": list_outputs[1],
                   "targets_l2": list_outputs[2]}

    return feature_map


def process_raw_feature(seq_raw_features, dim_feature, num_context, downsample):
    # 1-D, 2-D
    # if add_delta:
    #     seq_raw_features = tfAudio.add_delt(seq_raw_features)

    # Splice
    feature = tfAudio.splice(seq_raw_features,
                             left_num=0,
                             right_num=num_context)

    # downsample
    feature = tfAudio.down_sample(feature,
                                  rate=downsample,
                                  axis=0)

    dim_input = dim_feature * (num_context + 1)
    feature.set_shape([None, dim_input])

    return feature
