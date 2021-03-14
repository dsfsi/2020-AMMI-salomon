#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow==1.15 --user
# !pip install tensorflow-gpu==1.15 --user


# In[1]:


# # !pip install tensorflow==1.13.0rc1
# !pip install tensorflow==1.13.1
# !pip install tensorflow-gpu==1.13.1

# # !pip install ipdb
# # !pip install matplotlib


# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[4]:


# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[5]:


# get_ipython().system('nvidia-smi')


# In[6]:


# Imports 
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import glob
import six
import math
import sys
import time
import numpy as np
import argparse
import ipdb
import copy


import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import saver
from tensorflow.python.framework import function

import numpy as np
from random import shuffle
from tqdm import tqdm
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
from importlib import reload # not build in in py3
reload(sys)
from six import PY2

import scipy.io.wavfile as wav
from collections import Counter, defaultdict


# Dependency imports

# from utils import tfAudioTools as tfAudio#
# from utils.tfRecord import RESERVED_TOKENS_TO_INDEX
# from utils import text_reader
# from models import common_layers
# from models import common_attention
# from utils.beamsearch import *
# from utils import parallel
# from utils import datareader
# from models import common_hparams
# from models.transformer import Transformer


# 
# 
# # Colab configurations
# 

# In[7]:


# # link to google drive, click on the given link and choose the google drive account you would like to be available to you 

# from google.colab import drive
# drive.mount('/content/gdrive/')


# In[8]:


# get_ipython().system('pwd')


# In[9]:


# %cd ../content/gdrive/MyDrive/colab-ssh/2020-AMMI-salomon/notebooks/MT-ASR/WORKING_Tensorflow/synchronous/


# In[10]:


# get_ipython().system('ls ')


# # Utils

# In[11]:


# Reserved tokens for things like padding and EOS symbols.
PAD = "<PAD>"  # the index of PAD must be 0
EOS = "<EOS>"
L1 = "<2L1>"
L2 = "<2L2>"
DELAY = "<DELAY>"
UNK = "<UNK>"
SPACE = " "
RESERVED_TOKENS = [PAD, EOS, L1, L2, DELAY, UNK, SPACE]
RESERVED_TOKENS_TO_INDEX = {tok: idx for idx, tok in enumerate(RESERVED_TOKENS)}

# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7
L1_SYMBOL = RESERVED_TOKENS_TO_INDEX["<2L1>"]
L2_SYMBOL = RESERVED_TOKENS_TO_INDEX["<2L2>"]
DELAY_SYMBOL = RESERVED_TOKENS_TO_INDEX["<DELAY>"]

DELAY_SYMBOL = RESERVED_TOKENS_TO_INDEX["<DELAY>"]
L2_SYMBOL = RESERVED_TOKENS_TO_INDEX["<2L2>"]



PAD_idx = 0
EOS_idx = 1
L2R = 2
R2L = 3

# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7

# import ipdb

# Conversion between Unicode and UTF-8, if required (on Python2)
native_to_unicode = (lambda s: s.decode("utf-8")) if PY2 else (lambda s: s)
unicode_to_native = (lambda s: s.encode("utf-8")) if PY2 else (lambda s: s)


# Reserved tokens for things like padding and EOS symbols.
# PAD = "<PAD>"
# EOS = "<EOS>"
# L2R = "<2en>"
# R2L = "<2fr>"
# RESERVED_TOKENS = [PAD, EOS, L2R, R2L]
# if six.PY2:
#     RESERVED_TOKENS_BYTES = RESERVED_TOKENS
# else:
#     RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii"), bytes(L2R, "ascii"), bytes(R2L, "ascii")]
RESERVED_TOKENS_BYTES = RESERVED_TOKENS


# In[12]:


"""Implemetation of beam seach with penalties."""

def shape_list(x):
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def _merge_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.
        tensor: Tensor to reshape of shape [A, B, ...] --> [A*B, ...]
    """
    shape = shape_list(tensor)
    batch_size = shape[0]
    beam_size = shape[1]
    return tf.reshape(tensor, [batch_size * beam_size] + shape[2:])


def _unmerge_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].
        [batch_size*beam_size, ...] --> [batch_size, beam_size, ...]
    """
    shape = shape_list(tensor)
    new_shape = [batch_size] + [beam_size] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
        tensor: tensor to tile [batch_size, ...] --> [batch_size, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size
    return tf.tile(tensor, tile_dims)


def log_prob_from_logits(logits):
    return logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)


def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coodinate that contains the batch index for gathers.
    like [[0,0,0,0,],[1,1,1,1],..]
    """
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def beam_search(predict_next_symbols,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True):
    """Beam search with length penalties.
    """
    batch_size = shape_list(initial_ids)[0]
    initial_log_probs = tf.constant(
        [[0.] + (int(beam_size / 2) - 1) * [-float("inf")] + [0.] + (int(beam_size / 2) - 1) * [-float("inf")]])
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])  # (batch_size, beam_size)

    initial_ids_1 = L1_SYMBOL * tf.ones([batch_size, 1], dtype=tf.int32)  # index 2 == <l1>
    initial_ids_2 = DELAY_SYMBOL * tf.ones([batch_size, 1], dtype=tf.int32)  # index 3 == <l2>

    initial_ids_1 = tf.concat([initial_ids, initial_ids_1], axis=1)  # [batch, 2]
    initial_ids_2 = tf.concat([initial_ids, initial_ids_2], axis=1)

    alive_seq_1 = tf.tile(tf.expand_dims(initial_ids_1, 1),
                          [1, tf.cast(beam_size / 2, tf.int32), 1])  # [batch, beam/2, 2]
    alive_seq_2 = tf.tile(tf.expand_dims(initial_ids_2, 1), [1, tf.cast(beam_size / 2, tf.int32), 1])
    alive_seq = tf.concat([alive_seq_1, alive_seq_2], axis=1)  # [batch, beam, 2]

    states = nest.map_structure(
        lambda state: _expand_to_beam_size(state, beam_size), states)
    finished_seq_1 = tf.zeros(shape_list(alive_seq), tf.int32, name="fin-seq1")
    finished_seq_2 = tf.zeros(shape_list(alive_seq), tf.int32, name="fin-seq2")
    finished_scores_1 = tf.ones([batch_size, beam_size]) * -INF
    finished_scores_2 = tf.ones([batch_size, beam_size]) * -INF
    finished_flags_1 = tf.zeros([batch_size, beam_size], tf.bool, name="fin-flag1")
    finished_flags_2 = tf.zeros([batch_size, beam_size], tf.bool, name="fin-flag2")

    def _beam_search_step(i,
                          alive_seq,
                          alive_log_probs,
                          finished_seq_1,
                          finished_seq_2,
                          finished_scores_1,
                          finished_scores_2,
                          finished_flags_1,
                          finished_flags_2,
                          states):
        """Inner beam seach loop.
        """
        # 1. Get the current topk items.
        flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])
        flat_states = nest.map_structure(_merge_beam_dim, states)
        flat_logits, flat_states = predict_next_symbols(flat_ids, i, batch_size, beam_size, flat_states)  # !!
        states = nest.map_structure(
            lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_states)

        logits = tf.reshape(flat_logits, [batch_size, beam_size, -1])  # (batch, beam, vocab)
        candidate_log_probs = log_prob_from_logits(logits)  # softmax
        log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)
        length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), alpha)
        curr_scores = log_probs / length_penalty
        # flat_curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size]) # (batch, beam*vocab)
        flat_curr_scores = tf.reshape(curr_scores, [-1, 2, tf.cast(beam_size / 2,
                                                                   tf.int32) * vocab_size])  # [batch, 2, (beam/2) * vocab]
        # topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size * 2) # (batch, 2*beam)
        topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size)  # [batch, 2, beam]
        topk_log_probs = topk_scores * length_penalty
        topk_log_probs = tf.reshape(topk_log_probs, [-1, 2 * beam_size])  # add; [batch, 2*beam]
        topk_scores = tf.reshape(topk_scores, [-1, 2 * beam_size])
        topk_beam_index = topk_ids // vocab_size  # like [[0,1,1,0],[1,1,0,0],[1,0,0,0],...], e.g. beam=2
        topk_ids %= vocab_size  # Unflatten the ids

        topk_beam_index_1 = tf.concat([tf.expand_dims(topk_beam_index[:, 0, :], 1),
                                       tf.expand_dims(topk_beam_index[:, 1, :] + tf.cast(beam_size / 2, tf.int32), 1)],
                                      axis=1)
        topk_beam_index = tf.reshape(topk_beam_index_1, [-1, beam_size * 2])
        topk_ids = tf.reshape(topk_ids, [-1, beam_size * 2])

        batch_pos = compute_batch_indices(batch_size,
                                          beam_size * 2)
        # like [[0,0,0,0,],[1,1,1,1],[2,2,2,2],...] (batch, 2*beam)
        topk_coordinates = tf.stack([batch_pos, topk_beam_index],
                                    axis=2)
        # like [[[0,0],[0,1],[0,1],[0,0]], [[1,1],[1,1],[1,0],[1,0]], [[2,1],[2,0],[2,0],[2,0]],...]  (batch, 2*beam, 2)
        topk_seq = tf.gather_nd(alive_seq, topk_coordinates)  # (batch, 2*beam, lenght)
        states = nest.map_structure(
            lambda state: tf.gather_nd(state, topk_coordinates), states)
        topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)  # (batch, 2*beam, length+1)
        topk_finished = tf.equal(topk_ids, eos_id)  # (batch, 2*beam)

        # 2. Extract the ones that have finished and haven't finished
        curr_scores = topk_scores + tf.to_float(topk_finished) * -INF  # (batch, 2*beam)
        curr_scores = tf.reshape(curr_scores, [batch_size, 2, beam_size])
        _, topk_indexes = tf.nn.top_k(curr_scores, k=tf.cast(beam_size / 2, tf.int32))  # [batch, 2, beam/2]
        topk_indexes_tmp = topk_indexes[:, 1, :] + beam_size
        topk_indexes = tf.concat([tf.expand_dims(topk_indexes[:, 0, :], 1), tf.expand_dims(topk_indexes_tmp, 1)],
                                 axis=1)
        topk_indexes = tf.reshape(topk_indexes, [batch_size, beam_size])

        batch_pos_2 = compute_batch_indices(batch_size, beam_size)
        top_coordinates = tf.stack([batch_pos_2, topk_indexes], axis=2)  # (batch, beam, 2)
        alive_seq = tf.gather_nd(topk_seq, top_coordinates)
        alive_log_probs = tf.gather_nd(topk_log_probs, top_coordinates)
        alive_states = nest.map_structure(
            lambda state: tf.gather_nd(state, top_coordinates), states)

        # 3. Recompute the contents of finished based on scores.
        finished_seq_1 = tf.concat(
            [finished_seq_1,
             tf.zeros([batch_size, beam_size, 1], tf.int32)], axis=2)
        finished_seq_2 = tf.concat(
            [finished_seq_2,
             tf.zeros([batch_size, beam_size, 1], tf.int32)], axis=2)

        curr_scores = topk_scores + (1. - tf.to_float(topk_finished)) * -INF

        topk_seq_tmp_1 = tf.slice(topk_seq, [0, 0, 0], [batch_size, beam_size, -1])
        topk_seq_tmp_2 = tf.slice(topk_seq, [0, beam_size, 0], [batch_size, beam_size, -1])
        curr_finished_seq_1 = tf.concat([finished_seq_1, topk_seq_tmp_1], axis=1)
        curr_finished_seq_2 = tf.concat([finished_seq_2, topk_seq_tmp_2], axis=1)

        curr_scores_tmp_1 = tf.slice(curr_scores, [0, 0], [batch_size, beam_size])
        curr_scores_tmp_2 = tf.slice(curr_scores, [0, beam_size], [batch_size, beam_size])
        curr_finished_scores_1 = tf.concat([finished_scores_1, curr_scores_tmp_1], axis=1)
        curr_finished_scores_2 = tf.concat([finished_scores_2, curr_scores_tmp_2], axis=1)

        topk_finished_tmp_1 = tf.slice(topk_finished, [0, 0], [batch_size, beam_size])
        topk_finished_tmp_2 = tf.slice(topk_finished, [0, beam_size], [batch_size, beam_size])
        curr_finished_flags_1 = tf.concat([finished_flags_1, topk_finished_tmp_1], axis=1)
        curr_finished_flags_2 = tf.concat([finished_flags_2, topk_finished_tmp_2], axis=1)

        _, topk_indexes_tmp_1 = tf.nn.top_k(curr_finished_scores_1, k=beam_size)
        _, topk_indexes_tmp_2 = tf.nn.top_k(curr_finished_scores_2, k=beam_size)

        top_coordinates_tmp_1 = tf.stack([batch_pos_2, topk_indexes_tmp_1], axis=2)
        top_coordinates_tmp_2 = tf.stack([batch_pos_2, topk_indexes_tmp_2], axis=2)
        finished_seq_1 = tf.gather_nd(curr_finished_seq_1, top_coordinates_tmp_1)
        finished_seq_2 = tf.gather_nd(curr_finished_seq_2, top_coordinates_tmp_2)
        finished_flags_1 = tf.gather_nd(curr_finished_flags_1, top_coordinates_tmp_1)
        finished_flags_2 = tf.gather_nd(curr_finished_flags_2, top_coordinates_tmp_2)
        finished_scores_1 = tf.gather_nd(curr_finished_scores_1, top_coordinates_tmp_1)
        finished_scores_2 = tf.gather_nd(curr_finished_scores_2, top_coordinates_tmp_2)

        return (i + 1, alive_seq, alive_log_probs, finished_seq_1, finished_seq_2, finished_scores_1, finished_scores_2,
                finished_flags_1, finished_flags_2, alive_states)

    def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq_1, unused_finished_seq_2,
                     finished_scores_1, finished_scores_2, finished_flags_1, finished_flags_2, unused_states):
        """Checking termination condition.
        """
        if not stop_early:
            return tf.less(i, decode_length)
        max_length_penalty = tf.pow(((5. + tf.to_float(decode_length)) / 6.), alpha)
        lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty
        lowest_score_of_finished_1 = tf.reduce_min(
            finished_scores_1 * tf.to_float(finished_flags_1), axis=1)
        lowest_score_of_finished_2 = tf.reduce_min(
            finished_scores_2 * tf.to_float(finished_flags_2), axis=1)
        lowest_score_of_finished_1 += (
                (1. - tf.to_float(tf.reduce_any(finished_flags_1, 1))) * -INF)
        lowest_score_of_finished_2 += (
                (1. - tf.to_float(tf.reduce_any(finished_flags_2, 1))) * -INF)
        bound_is_met = tf.reduce_all(  # return True when lowest_score_of_finished > lower_bound_alive_scores
            tf.logical_and(tf.greater(lowest_score_of_finished_1, lower_bound_alive_scores),
                           tf.greater(lowest_score_of_finished_2, lower_bound_alive_scores)))

        return tf.logical_and(
            tf.less(i, decode_length), tf.logical_not(bound_is_met))

    (_, alive_seq, alive_log_probs, finished_seq_1, finished_seq_2, finished_scores_1, finished_scores_2,
     finished_flags_1, finished_flags_2, _) = tf.while_loop(
        _is_finished,  # termination when return False
        _beam_search_step, [
            tf.constant(0), alive_seq, alive_log_probs, finished_seq_1, finished_seq_2,
            finished_scores_1, finished_scores_2, finished_flags_1, finished_flags_2, states],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None, None]),
            alive_log_probs.get_shape(),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, None, None]),
            finished_scores_1.get_shape(),
            finished_scores_2.get_shape(),
            finished_flags_1.get_shape(),
            finished_flags_2.get_shape(),
            nest.map_structure(
                lambda tensor: get_state_shape_invariants(tensor), states)],
        parallel_iterations=1,
        back_prop=False)

    alive_seq.set_shape((None, beam_size, None))  # (batch, beam, length)
    finished_seq_1.set_shape((None, beam_size, None))
    finished_seq_2.set_shape((None, beam_size, None))

    finished_seq_1 = tf.where(
        tf.reduce_any(finished_flags_1, 1), finished_seq_1, alive_seq)
    finished_seq_2 = tf.where(
        tf.reduce_any(finished_flags_2, 1), finished_seq_2, alive_seq)
    finished_scores_1 = tf.where(
        tf.reduce_any(finished_flags_1, 1), finished_scores_1, alive_log_probs)
    finished_scores_2 = tf.where(
        tf.reduce_any(finished_flags_2, 1), finished_scores_2, alive_log_probs)
    return finished_seq_1, finished_seq_2, finished_scores_1, finished_scores_2


# In[13]:


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
                delay_target_l2 = tf.concat([tf.constant([DELAY_SYMBOL, DELAY_SYMBOL, L2_SYMBOL], tf.int32),
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

        rand_inputs, rand_target_l1, rand_target_l2 =             feature_map["inputs"], feature_map["targets_l1"], feature_map["targets_l2"]

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
    #     seq_raw_features = add_delt(seq_raw_features)

    # Splice
    feature = splice(seq_raw_features,
                             left_num=0,
                             right_num=num_context)

    # downsample
    feature = down_sample(feature,
                                  rate=downsample,
                                  axis=0)

    dim_input = dim_feature * (num_context + 1)
    feature.set_shape([None, dim_input])

    return feature


# In[14]:


PAD_idx = 0
EOS_idx = 1
L2R = 2
R2L = 3


def token_generator_three(source_path, target_path_l2r, target_path_r2l, token_vocab_src, token_vocab_tgt, eos=1, pad=1, l2r=1, r2l=1):
    """Generator for sequence-to-sequence tasks that uses tokens.
    """
    eos_list = [] if eos is None else [PAD_idx]
    pad_list = [] if pad is None else [EOS_idx]
    l2r_list = [] if l2r is None else [L2R]
    r2l_list = [] if r2l is None else [R2L]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path_l2r, mode="r") as target_file_l2r:
            with tf.gfile.GFile(target_path_r2l, mode="r") as target_file_r2l:

                source, target_l2r, target_r2l = source_file.readline(), target_file_l2r.readline(), target_file_r2l.readline()
                while source and target_l2r and target_r2l:
                    source_ints = token_vocab_src.encode(source.strip()) + eos_list 
        
                    t_l2r = token_vocab_tgt.encode(target_l2r.strip())
                    t_r2l = token_vocab_tgt.encode(target_r2l.strip())
                    print (len(t_l2r))
                    print (len(t_r2l))
                    t_length_max = max(len(t_l2r),len(t_r2l))
                    t_l2r_add_len = t_length_max - len(t_l2r)
                    t_r2l_add_len = t_length_max - len(t_r2l)
                    # let len(target_ints_l2r)==len(target_ints_r2l)
                    target_ints_l2r = l2r_list + t_l2r + t_l2r_add_len*pad_list + eos_list
                    target_ints_r2l = r2l_list + t_r2l + t_r2l_add_len*pad_list + eos_list

                    yield {"inputs": source_ints, "targets_l2r": target_ints_l2r, "targets_r2l": target_ints_r2l}
                    source, target_l2r, target_r2l = source_file.readline(), target_file_l2r.readline(), target_file_r2l.readline()


def translation_token_generator(data_dir, tmp_dir, train_src_name, train_tgt_name, vocab_src_name, vocab_tgt_name):
  
    train_src_path = os.path.join(tmp_dir, train_src_name)
    train_tgt_path_l2r = os.path.join(tmp_dir, train_tgt_name + ".l2r")
    train_tgt_path_r2l = os.path.join(tmp_dir, train_tgt_name + ".r2l")

    token_vocab_src_dir = os.path.join(data_dir, vocab_src_name)
    token_vocab_tgt_dir = os.path.join(data_dir, vocab_tgt_name)
    if not tf.gfile.Exists(token_vocab_src_dir):
        tf.gfile.Copy(os.path.join(tmp_dir, vocab_src_name), token_vocab_src_dir)
    if not tf.gfile.Exists(token_vocab_tgt_dir):
        tf.gfile.Copy(os.path.join(tmp_dir, vocab_tgt_name), token_vocab_tgt_dir)

    token_vocab_src = TokenTextEncoder(vocab_filename=token_vocab_src_dir)
    token_vocab_tgt = TokenTextEncoder(vocab_filename=token_vocab_tgt_dir)
    return token_generator_three(train_src_path, train_tgt_path_l2r, train_tgt_path_r2l, token_vocab_src, token_vocab_tgt, 1,1,1,1)


###########################################################

def to_example(dictionary):
    """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
    features = {}
    for (k, v) in six.iteritems(dictionary):
        if not v:
            raise ValueError("Empty generated field: %s", str((k, v)))
        if isinstance(v[0], six.integer_types):
            features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
        elif isinstance(v[0], float):
            features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
        elif isinstance(v[0], six.string_types):
            if not six.PY2:  # Convert in python 3.
                v = [bytes(x, "utf-8") for x in v]
            features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
        elif isinstance(v[0], bytes):
            features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
        else:
            raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                    (k, str(v[0]), str(type(v[0]))))
    return tf.train.Example(features=tf.train.Features(feature=features))


def generate_files(generator,
                   output_name,
                   output_dir,
                   num_shards=1,
                   max_cases=None):
    """Generate cases from a generator and save as TFRecord files.
    """
    writers = []
    output_files = []
    for shard in xrange(num_shards):
        output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        output_files.append(output_file)
        writers.append(tf.python_io.TFRecordWriter(output_file))

    counter, shard = 0, 0
    for case in generator:
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("Generating case %d for %s." % (counter, output_name))
        counter += 1
        if max_cases and counter > max_cases:
            break
        sequence_example = to_example(case)
        writers[shard].write(sequence_example.SerializeToString())
        shard = (shard + 1) % num_shards

    for writer in writers:
        writer.close()

    return output_files


def read_records(filename):
    reader = tf.python_io.tf_record_iterator(filename)
    records = []
    for record in reader:
        records.append(record)
    if len(records) % 100000 == 0:
        tf.logging.info("read: %d", len(records))
    return records


def write_records(records, out_filename):
    writer = tf.python_io.TFRecordWriter(out_filename)
    for count, record in enumerate(records):
        writer.write(record)
    if count > 0 and count % 100000 == 0:
        tf.logging.info("write: %d", count)
    writer.close()


# In[15]:


"""Implemetation of beam seach with penalties."""

def _beam_decode(features, decode_length, beam_size, top_beams, alpha, local_features):

    decoded_ids_l1, decoded_ids_l2, scores_l1, scores_l2 =         _fast_decode(features, decode_length, beam_size, top_beams, alpha, local_features)
    return {"outputs_l1": decoded_ids_l1, "outputs_l2": decoded_ids_l2,
            "scores_l1": scores_l1, "scores_l2": scores_l2}


def _fast_decode(features,
                 decode_length,
                 beam_size=1,
                 top_beams=1,
                 alpha=1.0,
                 local_features=None):
    """Fast decoding.
    """
    if local_features["_num_datashards"] != 1:
        raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = local_features["_data_parallelism"]
    hparams = local_features["_hparams"]

    inputs = features["inputs"]
    batch_size = tf.shape(features["inputs"])[0]
    target_modality = local_features["_hparams"].target_modality
    decode_length = tf.constant(decode_length)

    input_modality = local_features["_hparams"].input_modality
    with tf.variable_scope(input_modality.name):
        inputs = local_features["_shard_features"]({"inputs": inputs})["inputs"]
    with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
                local_features["encode"], inputs, hparams)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if hparams.pos == "timing":
        timing_signal = get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
        """Performs preprocessing steps on the targets to prepare for the decoder.
        Returns: Processed targets [batch_size, 1, hidden_dim]
        """
        # _shard_features called to ensure that the variable names match
        targets = local_features["_shard_features"]({"targets": targets})["targets"]
        with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
        targets = flatten4d3d(targets)

        # TODO(llion): Explain! Is this even needed?
        targets = tf.cond(
            tf.equal(i, 0), lambda: tf.concat([tf.zeros_like(targets)[:,:1,:],targets[:,1:,:]], axis=1), lambda: targets)
        
        if hparams.pos == "timing":
            timing_signal_1 = tf.cond(
                        tf.equal(i, 0), lambda: timing_signal[:, i:i + 2], lambda: timing_signal[:, i+1:i + 2])
            targets += timing_signal_1
        return targets

    decoder_self_attention_bias = (
            attention_bias_lower_triangle(decode_length+1))

    def predict_next_symbols(ids, i, batch_size, beam_size, cache):
        """Go from ids to logits for next symbol."""
        ids = tf.cond(
                    tf.equal(i, 0), lambda: ids[:, -2:], lambda: ids[:, -1:])
        targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
        targets = preprocess_targets(targets, i)

        bias_1 = decoder_self_attention_bias[:, :, i:i + 2, :i + 2]
        bias_2 = decoder_self_attention_bias[:, :, i+1:i + 2, :i + 2]
        bias = tf.cond(
                    tf.equal(i, 0), lambda: bias_1, lambda: bias_2)
        
        s = tf.shape(cache['encoder_output'])
        cache['encoder_output'] = tf.reshape(cache['encoder_output'],[s[0],s[1],hparams.hidden_size])
        with tf.variable_scope("body"):
            body_outputs = dp(
                local_features["decode"], targets, cache["encoder_output"],
                    cache["encoder_decoder_attention_bias"], bias, hparams, batch_size, beam_size, cache)

        with tf.variable_scope(target_modality.name):
            logits = target_modality.top_sharded(body_outputs, None, dp)[0]
            
        tf.logging.info("logits's shape is {0}".format(logits[0].shape))
        return tf.squeeze(logits, axis=[0, 3])[:, -1, :], cache

    key_channels = hparams.hidden_size
    value_channels = hparams.hidden_size
    num_layers = hparams.num_hidden_layers

    cache = {
        "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, key_channels]),
                "v": tf.zeros([batch_size, 0, value_channels]),
        }
        for layer in range(num_layers)
    }

    for layer in cache:
        cache[layer]["k"].set_shape = tf.TensorShape([None, None, key_channels])
        cache[layer]["v"].set_shape = tf.TensorShape([None, None, value_channels])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output

    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    target_modality = (
            local_features["_hparams"].target_modality)
    vocab_size = target_modality.top_dimensionality
    initial_ids = tf.zeros([batch_size, 1], dtype=tf.int32)
    decoded_ids_l1, decoded_ids_l2, scores_l1, scores_l2 = beam_search(
            predict_next_symbols,
            initial_ids,
            beam_size,
            decode_length,
            vocab_size,
            alpha,
            states=cache,
            stop_early=(top_beams == 1))

    if top_beams == 1:
        decoded_ids_l1 = decoded_ids_l1[:, 0, 1:]
        decoded_ids_l2 = decoded_ids_l2[:, 0, 1:]
    else:
        decoded_ids_l1 = decoded_ids_l1[:, :top_beams, 1:]
        decoded_ids_l2 = decoded_ids_l2[:, :top_beams, 1:]

    return decoded_ids_l1, decoded_ids_l2, scores_l1, scores_l2


# In[16]:


"""Utilities for creating Sparsely-Gated Mixture-of-Experts Layers.

See the most recent draft of our ICLR paper:
https://openreview.net/pdf?id=B1ckMDqlg
"""

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def ConvertGradientToTensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `ConvertGradientToTensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


class Parallelism(object):
  """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in xrange(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=None,
               caching_devices=None,
               daisy_chain_variables=False):
    """Create a Parallelism.

    Args:
      device_names_or_functions: A list of of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.

    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._MaybeRepeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables

  def __call__(self, fn, *args, **kwargs):
    """A parallel set of function calls (using the specified devices).

    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.

    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
    # Construct lists or args and kwargs for each function.
    if args:
      my_args = TransposeListOfLists([self._MaybeRepeat(arg) for arg in args])
    else:
      my_args = [[] for _ in xrange(self.n)]
    my_kwargs = [{} for _ in xrange(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._MaybeRepeat(v)
      for i in xrange(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._MaybeRepeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    for i in xrange(self.n):

      def DaisyChainGetter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          v = tf.identity(cache[name])
        else:
          var = getter(name, *args, **kwargs)
          v = tf.identity(var._ref())  # pylint: disable=protected-access
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def CachingGetter(getter, name, *args, **kwargs):
        v = getter(name, *args, **kwargs)
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]
        with tf.device(self._caching_devices[i]):
          ret = tf.identity(v._ref())  # pylint: disable=protected-access
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = DaisyChainGetter
      elif self._caching_devices:
        custom_getter = CachingGetter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope('parallel_%d' % i):
        with tf.variable_scope(
            tf.get_variable_scope(),
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          with tf.device(self._devices[i]):
            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
    if isinstance(outputs[0], tuple):
      outputs = list(zip(*outputs))
      outputs = tuple([list(o) for o in outputs])
    return outputs

  @property
  def n(self):
    return self._n

  @property
  def devices(self):
    return self._devices

  def _MaybeRepeat(self, x):
    """Utility function for processing arguments that are singletons or lists.

    Args:
      x: either a list of self.n elements, or not a list.

    Returns:
      a list of self.n elements.
    """
    if isinstance(x, list):
      assert len(x) == self.n
      return x
    else:
      return [x] * self.n


def Parallel(device_names_or_functions, fn, *args):
  """Deprecated interface.

  Use `Parallelism(device_names_or_functions)(fn, *args)` instead.

  Args:
    device_names_or_functions: A list of length n.
    fn: a function or a list of n functions.
    *args: additional args.  Each arg should either be not a list, or a list
       of length n.

  Returns:
    either a single list of length n (if fn does not return a tuple), or a
    tuple of lists of length n (if fn returns a tuple).
  """
  return Parallelism(device_names_or_functions)(fn, *args)


def TransposeListOfLists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, 'cannot pass the empty list'
  return [list(x) for x in zip(*lol)]


# In[17]:


"""Encoders for text data.

* TextEncoder: base class
* ByteTextEncoder: for ascii text
* TokenTextEncoder: with user-supplied vocabulary file
* SubwordTextEncoder: invertible
"""

class TextEncoder(object):
    """Base class for converting from ints to/from human readable strings."""

    def __init__(self, num_reserved_ids=4):
        self._num_reserved_ids = len(RESERVED_TOKENS)

    def encode(self, s):

        """Transform a human-readable string into a sequence of int ids."""
        return [int(w) + self._num_reserved_ids for w in s.split()]

    def decode(self, ids):
        """Transform a sequence of int ids into a human-readable string."""

        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < self._num_reserved_ids:
                decoded_ids.append(RESERVED_TOKENS[int(id_)])
            else:
                decoded_ids.append(id_ - self._num_reserved_ids)
        return " ".join([str(d) for d in decoded_ids])

    @property
    def vocab_size(self):
        raise NotImplementedError()


class TokenTextEncoder(TextEncoder):
    """Encoder based on a user-supplied vocabulary."""

    def __init__(self, vocab_filename, reverse=False, num_reserved_ids=4):
        """Initialize from a file, one token per line."""
        super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._load_vocab_from_file(vocab_filename)

    def encode(self, sentence, replace_oov=None):
        """Converts a space-separated string of tokens to a list of ids."""
        tokens = sentence.strip().split()
        if replace_oov is not None:
            tokens = [t if t in self._token_to_id else replace_oov for t in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return " ".join([self._safe_id_to_token(i) for i in seq])

    @property
    def vocab_size(self):
        return len(self._id_to_token)

    def _safe_id_to_token(self, idx):
        return self._id_to_token.get(idx, "ID_%d" % idx)

    def _load_vocab_from_file(self, filename):
        """Load vocab from a file."""
        self._token_to_id = {}
        self._id_to_token = {}

        for idx, tok in enumerate(RESERVED_TOKENS):
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok

        token_start_idx = self._num_reserved_ids
        # token_start_idx = 0
        with tf.gfile.Open(filename) as f:
            
            vocab = [line.strip().split() for line in f][0]
            # TODO I need to modify vocabulary size automatically
            vocab = vocab[:30000] if 30000 else vocab
            
            new_idx = token_start_idx -1
            for _, tok in enumerate(vocab):

                if tok not in self._token_to_id:
                    new_idx += 1
                    self._token_to_id[tok] = new_idx
                    self._id_to_token[new_idx] = tok
            
#             for i, line in enumerate(f):
#                 idx = token_start_idx + i
#                 tok = line.strip().split()[0]
#                 ipdb.set_trace()
#                 self._token_to_id[tok] = idx
#                 self._id_to_token[idx] = tok
                
            
            assert len(self._token_to_id) == len(self._id_to_token)
            print('vocab size is %d' % len(self._token_to_id))


#     def _load_vocab_from_file(self, vocab_path, vocab_size=None):
# #         vocab = [line.strip().split()[0] for line in open(vocab_path, 'r')]
#         vocab = [line.strip().split() for line in open(vocab_path, 'r')][0]
#         vocab = vocab[:vocab_size] if vocab_size else vocab
#         token2idx = defaultdict()
#         idx2token = defaultdict()

#         for idx, tok in enumerate(RESERVED_TOKENS):
#             token2idx[tok] = idx
#             idx2token[idx] = tok
#         token_start_idx = self._num_reserved_ids
#         new_idx = token_start_idx -1
#         for _, tok in enumerate(vocab):
        
#             if tok not in token2idx:
#                 new_idx += 1
#                 token2idx[tok] = new_idx
#                 idx2token[new_idx] = tok
                
# #         for idx, tok in enumerate(vocab):
# #             new_idx = token_start_idx + idx
# #             token2idx[tok] = new_idx
# #             idx2token[new_idx] = tok

# #         ipdb.set_trace()
#         assert len(token2idx) == len(idx2token)
#         print('vocab size is %d' % len(token2idx))
#         return token2idx, idx2token
###########################################################
def examples_queue(data_sources,
                   data_fields_to_features,
                   training,
                   capacity=32,
                   data_items_to_decoders=None,
                   data_items_to_decode=None):
    """Contruct a queue of training or evaluation examples.
    """
    with tf.name_scope("examples_queue"):
        # Read serialized examples using slim parallel_reader.
        num_epochs = None if training else 1
        data_files = tf.contrib.slim.parallel_reader.get_data_files(data_sources)
        num_readers = min(4 if training else 1, len(data_files))
        _, example_serialized = tf.contrib.slim.parallel_reader.parallel_read(
            data_sources,
            tf.TFRecordReader,
            num_epochs=num_epochs,
            shuffle=training,
            capacity=2 * capacity,
            min_after_dequeue=capacity,
            num_readers=num_readers)

        if data_items_to_decoders is None:
            data_items_to_decoders = {
            field: tf.contrib.slim.tfexample_decoder.Tensor(field)
            for field in data_fields_to_features
        }

        decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
            data_fields_to_features, data_items_to_decoders)

        if data_items_to_decode is None:
            data_items_to_decode = list(data_items_to_decoders)

        decoded = decoder.decode(example_serialized, items=data_items_to_decode)
        return {
            field: tensor
            for (field, tensor) in zip(data_items_to_decode, decoded)
        }


def input_pipeline(data_file_pattern, capacity, mode):
    """Input pipeline, returns a dictionary of tensors from queues."""

    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        #"targets_l2r": tf.VarLenFeature(tf.int64)}
        "targets_l2r": tf.VarLenFeature(tf.int64),
        "targets_r2l": tf.VarLenFeature(tf.int64)}
    data_items_to_decoders = None

    examples = examples_queue(
        [data_file_pattern],
        data_fields,
        training=(mode == tf.contrib.learn.ModeKeys.TRAIN),
        capacity=capacity,
        data_items_to_decoders=data_items_to_decoders)

    # We do not want int64s as they do are not supported on GPUs.
    return {k: tf.to_int32(v) for (k, v) in six.iteritems(examples)}


def batch_examples(examples, batching_scheme):
    """Given a queue of examples, create batches of examples with similar lengths.
    """
    with tf.name_scope("batch_examples"):
        # The queue to bucket on will be chosen based on maximum length.
        max_length = 0
        for v in examples.values():
        # For images the sequence length is the size of the spatial dimensions.
            sequence_length = (tf.shape(v)[0] if len(v.get_shape()) < 3 else
                    tf.shape(v)[0] * tf.shape(v)[1])
            max_length = tf.maximum(max_length, sequence_length)
        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_length,
            examples,
            batching_scheme["batch_sizes"],
            [b + 1 for b in batching_scheme["boundaries"]],
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=[2 * b for b in batching_scheme["batch_sizes"]],
            dynamic_pad=True,
            keep_input=(max_length <= batching_scheme["max_length"]))
        return outputs


def bucket_boundaries(max_length, min_length=8, mantissa_bits=2):
    """A default set of length-bucket boundaries."""
    x = min_length
    boundaries = []
    while x < max_length:
        boundaries.append(x)
        x += 2**max(0, int(math.log(x, 2)) - mantissa_bits)
    return boundaries


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
    """A batching scheme based on model hyperparameters.
    """
    max_length = hparams.max_length or hparams.batch_size
    boundaries = bucket_boundaries(
        max_length, mantissa_bits=hparams.batching_mantissa_bits)
    batch_sizes = [
        max(1, hparams.batch_size // length)
        for length in boundaries + [max_length]
    ]
    batch_sizes = [b * shard_multiplier for b in batch_sizes]
    max_length *= length_multiplier
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    return {
        "boundaries": boundaries,
        "batch_sizes": batch_sizes,
        "max_length": (max_length if drop_long_sequences else 10**9)
    }


def get_datasets(data_dir, mode):
    """Return the location of a dataset for a given mode."""
    datasets = []
    for problem in ["translation", ]:
    # for problem in ["wmt_ende_bpe32k", ]:
        # problem, _, _ = common_hparams.parse_problem_name(problem)
        path = os.path.join(data_dir, problem)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            datasets.append("%s-train*" % path)
        else:
            datasets.append("%s-dev*" % path)
    return datasets


# In[18]:


# tf fea opr
def tf_kaldi_fea_delt1(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l2 = tf.slice(feats_padded, [0, 0], shape)
    l1 = tf.slice(feats_padded, [1, 0], shape)
    r1 = tf.slice(feats_padded, [3, 0], shape)
    r2 = tf.slice(feats_padded, [4, 0], shape)

    delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2
    return delt1


def tf_kaldi_fea_delt2(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l4 = tf.slice(feats_padded, [0, 0], shape)
    l3 = tf.slice(feats_padded, [1, 0], shape)
    l2 = tf.slice(feats_padded, [2, 0], shape)
    l1 = tf.slice(feats_padded, [3, 0], shape)
    c = tf.slice(feats_padded, [4, 0], shape)
    r1 = tf.slice(feats_padded, [5, 0], shape)
    r2 = tf.slice(feats_padded, [6, 0], shape)
    r3 = tf.slice(feats_padded, [7, 0], shape)
    r4 = tf.slice(feats_padded, [8, 0], shape)

    delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)
    return delt2


def add_delt(feature):
    fb = []
    fb.append(feature)
    delt1 = tf_kaldi_fea_delt1(feature)
    fb.append(delt1)
    delt2 = tf_kaldi_fea_delt2(feature)
    fb.append(delt2)
    return tf.concat(axis=1, values=fb)


def cmvn_global(feature, mean, var):
    fea = (feature - mean) / var
    return fea


def cmvn_utt(feature):
    fea_mean = tf.reduce_mean(feature, 0)
    fea_var = tf.reduce_mean(tf.square(feature), 0)
    fea_var = fea_var - fea_mean * fea_mean
    fea_ivar = tf.rsqrt(fea_var + 1E-12)
    fea = (feature - fea_mean) * fea_ivar
    return fea


def splice(features, left_num, right_num):
    """
    [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]]
    left_num=0, right_num=2:
        [[1 1 1 2 2 2 3 3 3]
         [2 2 2 3 3 3 4 4 4]
         [3 3 3 4 4 4 5 5 5]
         [4 4 4 5 5 5 6 6 6]
         [5 5 5 6 6 6 7 7 7]
         [6 6 6 7 7 7 0 0 0]
         [7 7 7 0 0 0 0 0 0]]
    """
    shape = tf.shape(features)
    splices = []
    pp = tf.pad(features, [[left_num, right_num], [0, 0]])
    for i in range(left_num + right_num + 1):
        splices.append(tf.slice(pp, [i, 0], shape))
    splices = tf.concat(axis=1, values=splices)

    return splices


def down_sample(features, rate, axis=1):
    """
    features: batch x time x deep
    Notation: you need to set the shape of the output! tensor.set_shape(None, dim_input)
    """
    len_seq = tf.shape(features)[axis]

    return tf.gather(features, tf.range(len_seq, delta=rate), axis=axis)


def target_delay(features, num_target_delay):
    seq_len = tf.shape(features)[0]
    feats_part1 = tf.slice(features, [num_target_delay, 0], [seq_len-num_target_delay, -1])
    frame_last = tf.slice(features, [seq_len-1, 0], [1, -1])
    feats_part2 = tf.concat([frame_last for _ in range(num_target_delay)], axis=0)
    features = tf.concat([feats_part1, feats_part2], axis=0)

    return features


# In[19]:


# def get_argument():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--tmp_dir", default='./', help="Temporary storage directory.")
#     parser.add_argument("--data_dir", default='./', help="Directory with training data.")
    
#     parser.add_argument("--train_csv_name", default='./', help="Filename of training data.")
#     parser.add_argument("--dev_csv_name", default='./', help="Filename of dev data.")
#     parser.add_argument("--test_csv_name", default='./', help="Filename of test data.")
    
#     parser.add_argument("--wav_dir_train", default='./', help="Wavefile path of training data.")
#     parser.add_argument("--wav_dir_dev", default='./', help="Wavefile path of dev data.")
#     parser.add_argument("--wav_dir_test", default='./', help="Wavefile path of test data.")
    
#     parser.add_argument("--vocabA_name", default='./', help="Vocab language A file name.")
#     parser.add_argument("--vocabB_name", default='./', help="Vocab language B file name.")
    
#     parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size.")
    
#     parser.add_argument("-d", "--dim_raw_input", type=int, default=80, help="The dimension of input feature.")
#     args = parser.parse_args()
#     return args





def save2tfrecord(dataset, mode, dir_save, size_file=5000000):
    """
    Args:
        dataset = ASRdataSet(dataset, mode, dir_save, size_file)
        mode: Train or Dev
        dir_save: the dir to save the tfdata files
        size_file: average size of each record file
    Return:
        a folder consist of `tfdata.info`, `*.record`
    """

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_token = 0
    idx_file = -1
    num_damaged_sample = 0
    dim_feature = 0

    for i, sample in enumerate(tqdm(dataset)):
        if sample['inputs'] is None or not sample:
            num_damaged_sample += 1
            continue
        assert len(sample) == 4

        dim_feature = sample['inputs'].shape[-1]
        if (num_token // size_file) > idx_file:
            idx_file = num_token // size_file
            print('saving to file {}/{}.{}.record'.format(dir_save, mode, idx_file))
            writer = tf.python_io.TFRecordWriter('{}/{}.{}.record'.format(dir_save, mode, idx_file))

        example = tf.train.Example(
            features=tf.train.Features(
                feature={'inputs': _bytes_feature(sample['inputs'].tostring()),
                         'target_l1': _bytes_feature(sample['target_l1'].tostring()),
                         'target_l2': _bytes_feature(sample['target_l2'].tostring())}
            )
        )
        writer.write(example.SerializeToString())
        num_token += len(sample['inputs'])

    with open(os.path.join(dir_save, '%s.info' % mode), 'w') as fw:
        fw.write('data_file {}\n'.format(dataset.list_files))
        fw.write('dim_feature {}\n'.format(dim_feature))
        fw.write('num_tokens {}\n'.format(num_token))
        fw.write('size_dataset {}\n'.format(i-num_damaged_sample+1))
        fw.write('damaged samples: {}\n'.format(num_damaged_sample))

    return


class DataSet(object):
    def __init__(self, filepath, vocabA_path, vocabB_path, wav_path, dim_raw_input, vocab_size, _shuffle=False):
        self.list_utterances = self.gen_utter_list(filepath)
        self.list_files = filepath
        
        if _shuffle:
            self.shuffle_utts()
            
        self.wav_path = wav_path
        self.dim_raw_input = dim_raw_input
        self._num_reserved_ids = len(RESERVED_TOKENS)
        
        self.token2idxA, self.idx2tokenA = self._load_vocab_from_file(vocabA_path, vocab_size)
        self.token2idxB, self.idx2tokenB = self._load_vocab_from_file(vocabB_path, vocab_size)
        
        self.end_id = self.token2idxA['<EOS>']
        self.id_l1 = self.token2idxA['<2L1>']
        self.id_l2 = self.token2idxB['<2L2>']

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        # if len(utterance.strip().split('\t'))!=3:
        #     ipdb.set_trace()
        wavname, target_l1, target_l2 = utterance.strip().split('\t')
#         wavname, target_l1, target_l2 = utterance.strip().split('\t\t')
        try:
            wavid = wavname.split("_")[0]
#             print(utterance.strip().split('\t'))
#             ipdb.set_trace()
            feature = audio2vector(os.path.join(self.wav_path, wavname), self.dim_raw_input)
#             feature = audio2vector(os.path.join(self.wav_path, wavid, wavname), self.dim_raw_input)
        except:
            print("wavefile {} is empty or damaged, we pass it away.".format(wavname))
            feature = None
        target_l1 = np.array([self.id_l1] +
                             [self.token2idxA.get(word, self.token2idxA['<UNK>']) for word in target_l1.split(' ')] +
                             [self.end_id],
                             dtype=np.int32)

        target_l2 = np.array([self.id_l2] +
                             [self.token2idxB.get(word, self.token2idxB['<UNK>']) for word in target_l2.split(' ')] +
                             [self.end_id],
                             dtype=np.int32)
                             
        sample = {'id': wavname, 'inputs': feature, 'target_l1': target_l1, 'target_l2': target_l2}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        with open(list_files, 'r') as f:
            list_utter = f.readlines()
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)

    def __iter__(self):
        """
        utility the __getitem__ to impliment the __iter__
        """
        for idx in range(len(self)):
            yield self[idx]

    def __call__(self, idx):
        return self.__getitem__(idx)

    def _load_vocab_from_file(self, vocab_path, vocab_size=None):
#         vocab = [line.strip().split()[0] for line in open(vocab_path, 'r')]
        vocab = [line.strip().split() for line in open(vocab_path, 'r')][0]
        vocab = vocab[:vocab_size] if vocab_size else vocab
        token2idx = defaultdict()
        idx2token = defaultdict()

        for idx, tok in enumerate(RESERVED_TOKENS):
            token2idx[tok] = idx
            idx2token[idx] = tok
        token_start_idx = self._num_reserved_ids
        new_idx = token_start_idx -1
        for _, tok in enumerate(vocab):
        
            if tok not in token2idx:
                new_idx += 1
                token2idx[tok] = new_idx
                idx2token[new_idx] = tok
                
#         for idx, tok in enumerate(vocab):
#             new_idx = token_start_idx + idx
#             token2idx[tok] = new_idx
#             idx2token[new_idx] = tok

#         ipdb.set_trace()
        assert len(token2idx) == len(idx2token)
        print('vocab size is %d' % len(token2idx))
        return token2idx, idx2token


def audio2vector(audio_filename, dim_feature):
    '''
    Turn an audio file into feature representation.
    16k wav, size 283K -> len 903
    '''
    if audio_filename.endswith('.wav'):
        rate, sig = wav.read(audio_filename)
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(audio_filename))
    # Get fbank coefficients. numcep is the feature size
#     orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature).astype(np.float32)
    orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature, nfft=600).astype(np.float32)
    orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)

    return orig_inputs


# if __name__ == '__main__':
#     args = get_argument()
#     wav_path_train = args.wav_dir_train
#     wav_path_dev = args.wav_dir_dev
#     wav_path_test = args.wav_dir_test

#     vocabA_path = os.path.join(args.tmp_dir, args.vocabA_name)
#     vocabB_path = os.path.join(args.tmp_dir, args.vocabB_name)

#     train_csv_path = os.path.join(args.tmp_dir, args.train_csv_name)
#     dev_csv_path = os.path.join(args.tmp_dir, args.dev_csv_name)
#     test_csv_path = os.path.join(args.tmp_dir, args.test_csv_name)
#     dim_raw_input = args.dim_raw_input
#     vocab_size = args.vocab_size

#     dataset_dev = DataSet(dev_csv_path, vocabA_path, vocabB_path, wav_path_dev,
#                           dim_raw_input, vocab_size, _shuffle=False)
#     save2tfrecord(dataset_dev, 'dev', args.data_dir)
    
#     dataset_test = DataSet(test_csv_path, vocabA_path, vocabB_path, wav_path_test,
#                            dim_raw_input, vocab_size, _shuffle=False)
#     save2tfrecord(dataset_test, 'test', args.data_dir)
    
#     dataset_train = DataSet(train_csv_path, vocabA_path, vocabB_path, wav_path_train,
#                             dim_raw_input, vocab_size, _shuffle=True)
#     save2tfrecord(dataset_train, 'train', args.data_dir)


# In[20]:


"""Utilities for trainer binary."""


def create_hparams():
    """Returns hyperparameters, including any flag value overrides.
    """
    if FLAGS.hparams_set == "transformer_params_base":
        hparams = transformer_params_base(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) ## !!
    elif FLAGS.hparams_set == "transformer_params_big":
        hparams = transformer_params_big(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) ## !!
    elif FLAGS.hparams_set == "transformer_params_small":
        hparams = transformer_params_small(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name)  ## !!
#     elif FLAGS.hparams_set == "transformer_params_listra":
#         hparams = transformer_params_listra(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name)  ## !!
    else:
        raise ValueError("Do not have right model params")

    hparams.vocab_src_size = FLAGS.vocab_src_size
    hparams.vocab_tgt_size = FLAGS.vocab_tgt_size

    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)

    hparams.add_hparam("data_dir", FLAGS.data_dir)
    hparams.add_hparam("dim_feature", FLAGS.dim_feature)
    hparams.add_hparam("num_context", FLAGS.num_context)
    hparams.add_hparam("downsample", FLAGS.downsample)

    return hparams


def run(model, output_dir):
    """Runs an Estimator locally or distributed.
    """
    # Build Params
    tf.logging.info("Build Params...")
    hparams = create_hparams()

    if FLAGS.train_steps == 0:
        tf.logging.info("Prepare for Inference...")
        inference_run(model, hparams, output_dir)
        return

    tf.logging.info("Prepare for Training...")
    train_run(model, hparams, output_dir)
    return


def train_run(model, hparams, output_dir):
    # Build Data
    tf.logging.info("Build Data...")
    train_input_fn = get_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        hparams=hparams)
    
    # Build Model
    tf.logging.info("Build Model...")
    model_fn = model_builder(model, hparams=hparams)
    
    # Build Graph
    tf.logging.info("Build Graph...")
    all_hooks = []
    with ops.Graph().as_default() as g:
        global_step = tf.train.create_global_step(g)
        features, labels = train_input_fn()
        model_fn_ops = model_fn(features, labels) # total_loss, train_op
        ops.add_to_collection(ops.GraphKeys.LOSSES, model_fn_ops[0])

        pre_saver = tf.train.Saver([var for var in tf.global_variables() if "encoder" in var.name])
        print(FLAGS.pretrain_output_dir)
        print(tf.train.latest_checkpoint(FLAGS.pretrain_output_dir))
        # ipdb.set_trace()

        saver = tf.train.Saver(sharded=True,
                               max_to_keep=FLAGS.keep_checkpoint_max,
                               defer_build=True,
                               save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
  
        all_hooks.extend([
            tf.train.StopAtStepHook(last_step=FLAGS.train_steps),
            tf.train.NanTensorHook(model_fn_ops[0]),
            tf.train.LoggingTensorHook(
            {
                'loss': model_fn_ops[0],
                'step': global_step
            },
            every_n_iter=100),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=output_dir,
                save_secs=FLAGS.save_checkpoint_secs or None,
                save_steps=FLAGS.save_checkpoint_steps or None,
                saver=saver) 
        ])

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=output_dir,
                hooks=all_hooks,
                save_checkpoint_secs=0,  # Saving is handled by a hook.
                config=session_config(gpu_mem_fraction=FLAGS.gpu_mem_fraction)) as mon_sess:
#             ipdb.set_trace()
            pre_saver.restore(mon_sess, tf.train.latest_checkpoint(FLAGS.pretrain_output_dir))
            loss = None
            while not mon_sess.should_stop():
                _, loss = mon_sess.run([model_fn_ops[1], model_fn_ops[0]])
        return loss


def _save_until_eos(hyp):
        ret = []
        index = 0
        # until you reach <EOS> id
        while index < len(hyp) and hyp[index] != 1:
            ret.append(hyp[index])
            index += 1
        return np.array(ret)

def inference_run(model, hparams, output_dir):

    # Build Model
    tf.logging.info("Build Model...")

    # Build Graph
    tf.logging.info("Build Graph...")
#     ipdb.set_trace()
    checkpoint_path = saver.latest_checkpoint(output_dir)
    if not checkpoint_path:
        raise LookupError("Couldn't find trained model at %s." % output_dir)

    dev_input_fn = get_input_fn(
        mode=tf.contrib.learn.ModeKeys.INFER,
        hparams=hparams)
    features, labels = dev_input_fn()

    model_fn_inference = model_builder_inference(model, hparams=hparams)
    infer_ops = model_fn_inference(features, labels) # predictions, None, None
    predictions, targets = infer_ops[0], infer_ops[1]
    mon_sess = tf.train.MonitoredSession(
        session_creator=tf.train.ChiefSessionCreator(
            checkpoint_filename_with_path=checkpoint_path,
            config=session_config(gpu_mem_fraction=FLAGS.gpu_mem_fraction)))

    targets_vocab = hparams.vocabulary["targets"]
    inputs_vocab = hparams.vocabulary["inputs"]

    inputs_l1 = []
    decodes_l1 = []
    scores_l1 = []
    inputs_l2 = []
    decodes_l2 = []
    scores_l2 = []

    start = time.clock()
    decode_num = 0
    with mon_sess as sess:
        while True:
            if mon_sess.should_stop():
                break
            preds, target = sess.run([predictions, targets])

            first_tensor = list(preds.values())[0]
            batch_length = first_tensor.shape[0]

            def log_fn(inputs_l1, inputs_l2, outputs_l1, outputs_l2):
                decoded_inputs_l1 = inputs_vocab.decode(_save_until_eos(inputs_l1.flatten())).replace("@@ ", "")
                tf.logging.info("INPUT en: %s" % decoded_inputs_l1)
                decoded_outputs_l1 = inputs_vocab.decode(
                    _save_until_eos(outputs_l1.flatten())).replace("@@ ", "")
                tf.logging.info("OUPUT en: %s" % decoded_outputs_l1)

                decoded_inputs_l2 = targets_vocab.decode(_save_until_eos(inputs_l2.flatten())).replace("@@ ", "")
                tf.logging.info("INPUT ln: %s" % decoded_inputs_l2)
                decoded_outputs_l2 = targets_vocab.decode(_save_until_eos(outputs_l2.flatten())).replace("@@ ", "")
                tf.logging.info("OUPUT ln: %s" % decoded_outputs_l2)
                return decoded_inputs_l1, decoded_inputs_l2, decoded_outputs_l1, decoded_outputs_l2

            for i in range(batch_length):
                decode_num += 1
                tf.logging.info("#########sentence {}#######".format(decode_num))
                result = {key: value[i] for key, value in six.iteritems(preds)}

                if FLAGS.decode_return_beams:
                    beam_decodes = []
                    output_l1_beams = np.split(
                        result["outputs_l1"], FLAGS.decode_beam_size, axis=0)
                    output_l2_beams = np.split(
                        result["outputs_l2"], FLAGS.decode_beam_size, axis=0)
                    index = 0
                    for output_l1, output_l2 in zip(output_l1_beams, output_l2_beams):
                        index += 1
                        tf.logging.info("##########beam {}########".format(index))
                        beam_decodes.append(log_fn(result["targets_l1"], result["targets_l2"], output_l1, output_l1))
                else:
                    input_text_1, input_text_2, output_text_1, output_text_2 = log_fn(result["targets_l1"],
                                                                                      result["targets_l2"],
                                                                                      result["outputs_l1"],
                                                                                      result["outputs_l2"])
                    inputs_l1.append(input_text_1)
                    inputs_l2.append(input_text_2)
                    decodes_l1.append(output_text_1)
                    decodes_l2.append(output_text_2)
                    scores_l1.append(result["scores_l1"])
                    scores_l2.append(result["scores_l2"])

    input_filename_l1 = os.path.join(FLAGS.output_dir, FLAGS.decode_to_file_l1) + ".ref"
    input_filename_l2 = os.path.join(FLAGS.output_dir, FLAGS.decode_to_file_l2) + ".ref"
    decode_filename_l1 = os.path.join(FLAGS.output_dir, FLAGS.decode_to_file_l1)
    decode_filename_l2 = os.path.join(FLAGS.output_dir, FLAGS.decode_to_file_l2)
    
    tf.logging.info("Writing decodes into %s" % decode_filename_l1)
    tf.logging.info("Writing decodes into %s" % decode_filename_l2)
    inputfile_l1 = tf.gfile.Open(input_filename_l1, "w")
    outfile_l1 = tf.gfile.Open(decode_filename_l1, "w")
    inputfile_l2 = tf.gfile.Open(input_filename_l2, "w")
    outfile_l2 = tf.gfile.Open(decode_filename_l2, "w")
    for index in range(len(decodes_l1)):
        inputfile_l1.write("%s\n" % inputs_l1[index])
        outfile_l1.write("%s\n" % (decodes_l1[index]))
        inputfile_l2.write("%s\n" % inputs_l2[index])
        outfile_l2.write("%s\n" % (decodes_l2[index]))

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)


def model_builder_inference(model, hparams):

    def model_fn(features, targets):

        mode = tf.contrib.learn.ModeKeys.INFER
        dp = data_parallelism()

        model_class = Transformer(hparams, mode, dp)

        result_list = model_class.infer(
            features,
            beam_size=FLAGS.decode_beam_size,
            top_beams=(FLAGS.decode_beam_size if FLAGS.decode_return_beams else 1),
            alpha=FLAGS.decode_alpha,
            decode_length=FLAGS.decode_extra_length)
        ret = {
            "outputs_l1": result_list["outputs_l1"],
            "outputs_l2": result_list["outputs_l2"],
            "scores_l1": result_list["scores_l1"],
            "scores_l2": result_list["scores_l2"]
        }, targets, None
        if "inputs" in features:
            ret[0]["inputs"] = features["inputs"]
            ret[0]["targets_l2"] = features["targets_l2"]
            ret[0]["targets_l1"] = targets
        if "infer_targets" in features:
            ret[0]["targets"] = features["infer_targets"]

        return ret

    return model_fn


def validate_flags():
    if not FLAGS.model:
        raise ValueError("Must specify a model with --model.")
    if not (FLAGS.hparams_set or FLAGS.hparams_range):
        raise ValueError("Must specify either --hparams_set or --hparams_range.")
    if not FLAGS.schedule:
        raise ValueError("Must specify --schedule.")
    if not FLAGS.output_dir:
        FLAGS.output_dir = "/tmp/tensor2tensor"
        tf.logging.warning("It is strongly recommended to specify --output_dir. "
                       "Using default output_dir=%s.", FLAGS.output_dir)


def session_config(gpu_mem_fraction=0.95):
    """The TensorFlow Session config to use."""
    graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)
    config = tf.ConfigProto(
        allow_soft_placement=True, graph_options=graph_options, gpu_options=gpu_options)

    return config


def model_builder(model, hparams):

    def model_fn(features, targets): 
        mode = tf.contrib.learn.ModeKeys.TRAIN
        features["targets_l1"] = targets
        dp = data_parallelism()
        tf.get_variable_scope().set_initializer(initializer(hparams))

        # We choose which problem to process.
        loss_moving_avgs = []  # Need loss moving averages for that.
        # for n in xrange(1):
        with tf.variable_scope("losses_avg"):
            loss_moving_avgs.append(
                tf.get_variable("total_loss", initializer=100.0, trainable=False))
            tf.get_variable("training_loss", initializer=100.0, trainable=False)
            tf.get_variable("extra_loss", initializer=100.0, trainable=False)

        def get_model():
            """Build the model for the n-th problem, plus some added variables."""
            model_class = Transformer(hparams, mode, dp) ##!!!!
            sharded_logits, training_loss, extra_loss = model_class.model_fn(features)

            with tf.variable_scope("losses_avg", reuse=True):
                loss_moving_avg = tf.get_variable("training_loss")
                o1 = loss_moving_avg.assign(loss_moving_avg * 0.9 + training_loss * 0.1)
                loss_moving_avg = tf.get_variable("extra_loss")
                o2 = loss_moving_avg.assign(loss_moving_avg * 0.9 + extra_loss * 0.1)
                loss_moving_avg = tf.get_variable("total_loss")
                total_loss = training_loss + extra_loss
                o3 = loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1)
            with tf.variable_scope("train_stats"):  # Count steps for this problem.
                problem_steps = tf.get_variable(
                    "steps", initializer=0, trainable=False)
                o4 = problem_steps.assign_add(1)
            with tf.control_dependencies([o1, o2, o3, o4]):  # Make sure the ops run.
                # Ensure the loss is a scalar here.
                total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")
            return [total_loss] + sharded_logits    # Need to flatten for cond later.

        result_list = get_model()
        sharded_logits, total_loss = result_list[1:], result_list[0]

        # Some training statistics.
        with tf.name_scope("training_stats"):
            learning_rate = hparams.learning_rate * learning_rate_decay(hparams)
            learning_rate /= math.sqrt(float(FLAGS.worker_replicas))
            tf.summary.scalar("learning_rate", learning_rate)
            global_step = tf.to_float(tf.train.get_global_step())

        # Log trainable weights and add decay.
        total_size, weight_decay_loss = 0, 0.0
        all_weights = {v.name: v for v in tf.trainable_variables()}
        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            v_size = int(np.prod(np.array(v.shape.as_list())))
            # tf.logging.info("Weight  %s\tshape    %s\tsize    %d",
            #        v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
            total_size += v_size
            if hparams.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
                # Add weight regularization if set and the weight is not a bias (dim>1).
                with tf.device(v._ref().device):  # pylint: disable=protected-access
                    v_loss = tf.nn.l2_loss(v) / v_size
                weight_decay_loss += v_loss
            is_body = len(v_name) > 5 and v_name[:5] == "body/"
            if hparams.weight_noise > 0.0 and is_body:
                # Add weight noise if set in hparams.
                with tf.device(v._ref().device):  # pylint: disable=protected-access
                    scale = learning_rate * 0.001
                    noise = tf.truncated_normal(v.shape) * hparams.weight_noise * scale
                    noise_op = v.assign_add(noise)
                with tf.control_dependencies([noise_op]):
                    total_loss = tf.identity(total_loss)
        tf.logging.info("Total trainable variables size: %d", total_size)
        if hparams.weight_decay > 0.0:
            total_loss += weight_decay_loss * hparams.weight_decay
        total_loss = tf.identity(total_loss, name="total_loss")

        # Define the train_op for the TRAIN mode.
        opt = _ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
        tf.logging.info("Computing gradients for global model_fn.")
        train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=total_loss,
            global_step=tf.train.get_global_step(),
            learning_rate=learning_rate,
            clip_gradients=hparams.clip_grad_norm or None,
            optimizer=opt,
            colocate_gradients_with_ops=True)
        tf.logging.info("Global model_fn finished.")
        return total_loss, train_op

    return model_fn


def initializer(hparams):
    if hparams.initializer == "orthogonal":
        return tf.orthogonal_initializer(gain=hparams.initializer_gain)
    elif hparams.initializer == "uniform":
        max_val = 0.1 * hparams.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif hparams.initializer == "normal_unit_scaling":
        return init_ops.variance_scaling_initializer(
            hparams.initializer_gain, mode="fan_avg", distribution="normal")
    elif hparams.initializer == "uniform_unit_scaling":
        return init_ops.variance_scaling_initializer(
            hparams.initializer_gain, mode="fan_avg", distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % hparams.initializer)


def learning_rate_decay(hparams):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(
        hparams.learning_rate_warmup_steps * FLAGS.worker_replicas)
    step = tf.to_float(tf.train.get_global_step())
    if hparams.learning_rate_decay_scheme == "noam":
        return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
            (step + 1) * warmup_steps**-1.5, (step + 1)**-0.5)
    else:
        raise ValueError("Unrecognized learning rate decay scheme: %s" %
                hparams.learning_rate_decay_scheme)


class _ConditionalOptimizer(tf.train.Optimizer):
    """Conditional optimizer."""

    def __init__(self, optimizer_name, lr, hparams):

        if optimizer_name == "Adam":
            # We change the default epsilon for Adam and re-scale lr.
            # Using LazyAdam as it's much faster for large vocabulary embeddings.
            self._opt = tf.contrib.opt.LazyAdamOptimizer(
                lr / 500.0,
                beta1=hparams.optimizer_adam_beta1,
                beta2=hparams.optimizer_adam_beta2,
                epsilon=hparams.optimizer_adam_epsilon)
        elif optimizer_name == "Momentum":
            self._opt = tf.train.MomentumOptimizer(
                lr, momentum=hparams.optimizer_momentum_momentum)
        else:
            self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

    def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
        return self._opt.compute_gradients(
                loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

    def apply_gradients(self, gradients, global_step=None, name=None):
        return self._opt.apply_gradients(
                gradients, global_step=global_step, name=name)


def _gpu_order(num_gpus):
    if FLAGS.gpu_order:
        ret = [int(s) for s in FLAGS.gpu_order.split(" ")]
        if len(ret) == num_gpus:
            return ret
    return list(range(num_gpus))


def data_parallelism(all_workers=False):
    """Over which devices do we split each training batch.
    """

    if FLAGS.schedule == "local_run":
        #assert not FLAGS.sync
        datashard_devices = ["gpu:%d" % d for d in _gpu_order(FLAGS.worker_gpu)]
        if FLAGS.locally_shard_to_cpu:
            datashard_devices += ["cpu:0"]
        caching_devices = None
    
    tf.logging.info("datashard_devices: %s", datashard_devices)
    tf.logging.info("caching_devices: %s", caching_devices)
    return Parallelism(
            datashard_devices,
            reuse=True,
            caching_devices=caching_devices,
            daisy_chain_variables=FLAGS.daisy_chain_variables)


# # Speech features

# In[21]:


# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
# from __future__ import division
import numpy
from python_speech_features import sigproc
from scipy.fftpack import dct


def mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
         nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,
         winfunc=lambda x:numpy.ones((x,))):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat


def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          winfunc=lambda x:numpy.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    energy = numpy.sum(pspec,1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log

    return feat,energy

def logfbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,winlen=0.025,winstep=0.01,
        nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
        winfunc=lambda x:numpy.ones((x,))):
    """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = sigproc.powspec(frames,nfft)
    pspec = numpy.where(pspec == 0,numpy.finfo(float).eps,pspec) # if things are all zeros we get problems

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    R = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))

    return numpy.dot(pspec*R,fb.T) / feat

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


# In[22]:


# This file includes routines for basic signal processing including framing and computing power spectra.
# Author: James Lyons 2012
import decimal

import numpy
import math
import logging


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])


# # Model

# In[23]:


"""Utilities for attention."""

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return x + signal


def embedding_to_padding(emb):
    """Input embeddings -> is_padding.
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.equal(emb_sum, 0.0)


def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.
    """
    lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    ret = -1e9 * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
        input: [batch, memory_length], return: [batch, 1, 1, memory_length].
    """
    ret = tf.to_float(memory_padding) * -1e9
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    note: [..., m] --> [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    note: [..., a, b] --> [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).
    note: [batch, length, channels] -> [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def sb_split_heads(x, num_heads):
    return tf.transpose(split_last_dimension(x, num_heads), [0, 1, 3, 2, 4])


def combine_heads(x):
    """Inverse of split_heads.
    note: [batch, num_heads, length, channels / num_heads] -> [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def sb_combine_heads(x):
    return combine_last_two_dimensions(tf.transpose(x, [0, 1, 3, 2, 4]))


def shape_list(x):
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def residual_fn(x, y, hparams):
    return layer_norm(x + tf.nn.dropout(
            y, 1.0 - hparams.residual_dropout))


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        cache=None,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """
    with tf.variable_scope(
        name,
        default_name="multihead_attention",
        values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
            # self attention
            combined = conv1d(
                    query_antecedent,
                    total_key_depth * 2 + total_value_depth,
                    1,
                    name="qkv_transform")
            q, k, v = tf.split(
                    combined, [total_key_depth, total_key_depth, total_value_depth],
                    axis=2)
        else:
            q = conv1d(
                    query_antecedent, total_key_depth, 1, name="q_transform")
            combined = conv1d(
                    memory_antecedent,
                    total_key_depth + total_value_depth,
                    1,
                    name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                 "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = combine_heads(x)
        x = conv1d(x, output_depth, 1, name="output_transform")
        return x


def sb_dot_product_attention_for_decoding(q,
                          k,
                          v,
                          bias,
                          batch_size=None,
                          beam_size=None,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        # if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights, image_shapes)
        final_l2r = tf.matmul(weights, v)  # [batch*beam, num_heads, length_tmp, hidden_size/num_heads]

        # calculate final_r2l
        shape = shape_list(k)
        new_shape = [batch_size]+[2]+[tf.cast(beam_size/2,tf.int32)]+shape[1:]
        k_ = tf.reshape(k, new_shape)  # [batch, 2, beam/2, num_heads, length_tmp, hidden_size/num_heads]
        k_ = tf.reverse(k_,[1])
        v_ = tf.reshape(v, new_shape)
        v_ = tf.reverse(v_,[1])

        shape_ = shape_list(k_)
        new_shape_ = [batch_size*beam_size]+shape_[3:]
        k_ = tf.reshape(k_, new_shape_)  # [batch*beam, num_heads, length_tmp, hidden_size/num_heads]
        v_ = tf.reshape(v_, new_shape_)
        logits_ = tf.matmul(q, k_, transpose_b=True)
        logits_ += bias
        weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        final_r2l = tf.matmul(weights_, v_)

        final_all = final_l2r + 0.1 * final_r2l  # [batch*beam, num_heads, length_tmp, hidden_size/num_heads]
        return final_all


def sb_dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [2, batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        bias = tf.expand_dims(bias, axis=0)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        # if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights[0], image_shapes)
        final_l2r = tf.matmul(weights, v)  # [2, batch, num_heads, length, hidden_size/num_heads]

        # calculate final_r2l
        k_ = tf.reverse(k, [0])
        v_ = tf.reverse(v, [0])
        logits_ = tf.matmul(q, k_, transpose_b=True)
        logits_ += bias
        weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        final_r2l = tf.matmul(weights_, v_)

        final_all = final_l2r + 0.1 * final_r2l
        return final_all  # [2, batch, num_heads, length, hidden_size/num_heads]


def multi_dot_product_attention(q, k, v, bias,
                                language_num=1,
                                dropout_rate=0.0,
                                summaries=False,
                                image_shapes=None,
                                name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [n, batch, num_heads, query_length, memory_length]
        final_list = []
        bias = tf.expand_dims(bias, axis=0)
        for i in range(language_num):
            # [1, batch, num_heads, query_length, memory_length] *
            # [n, batch, num_heads, query_length, memory_length]
            temp_q = tf.expand_dims(q[i], axis=0)
            logits = tf.matmul(temp_q, k, transpose_b=True)
            logits += bias
            # weights?
            weights = tf.nn.softmax(logits, name="attention_weights_%d" % i)
            weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
            # final? [n, batch, num_heads, query_length, memory_length]
            final = tf.matmul(weights, v)
            # W = [n, 1]
            # [n, batch, num_heads, query_length, memory_length] ==>
            # [1, batch, num_heads, query_length, memory_length]
            final = tf.matmul(W, final)
            final_list.append(final)
        # final_list [n, batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        bias = tf.expand_dims(bias, axis=0)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        # if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights[0], image_shapes)
        final_l2r = tf.matmul(weights, v)  # [2, batch, num_heads, length, hidden_size/num_heads]

        # calculate final_r2l
        k_ = tf.reverse(k, [0])
        v_ = tf.reverse(v, [0])
        logits_ = tf.matmul(q, k_, transpose_b=True)
        logits_ += bias # modify err, logits --> logits_
        weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        final_r2l = tf.matmul(weights_, v_)

        final_all = final_l2r + 0.1 * final_r2l
        return final_all  # [2, batch, num_heads, length, hidden_size/num_heads]


def sb_multihead_attention( query_antecedent,
                            memory_antecedent,
                            bias,
                            total_key_depth,
                            total_value_depth,
                            output_depth,
                            num_heads,
                            dropout_rate,
                            cache=None,
                            summaries=False,
                            image_shapes=None,
                            name=None,
                            is_decoding=False):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """

    with tf.variable_scope(
                name,
                default_name="sb_multihead_attention",
                values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
        # self attention
            combined = sb_conv1d(
                query_antecedent,
                total_key_depth * 2 + total_value_depth,
                1,
                name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth], axis=3) ## 2-->3
        else:
            q = sb_conv1d(
                query_antecedent, total_key_depth, 1, name="q_transform")
            combined = conv1d(
                memory_antecedent,
                total_key_depth + total_value_depth,
                1,
                name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

            k = tf.concat([tf.expand_dims(k,0), tf.expand_dims(k,0)], axis=0) ## [2, batch, length, hidden_size]
            v = tf.concat([tf.expand_dims(v,0), tf.expand_dims(v,0)], axis=0)

        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                 "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = sb_split_heads(q, num_heads)
        k = sb_split_heads(k, num_heads)
        v = sb_split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        if memory_antecedent is None:  # decoder self attention (synchronous bidirectional att)
            x = sb_dot_product_attention(q, k, v, bias, dropout_rate, summaries,
                                         image_shapes)  # q: [2, num_heads, length_tmp, lenght]
        else:  # enc-dec attention
            x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = sb_combine_heads(x)
        x = sb_conv1d(x, output_depth, 1, name="output_transform")
        return x


def sb_multihead_attention_for_decoding(query_antecedent,
                                        memory_antecedent,
                                        bias,
                                        total_key_depth,
                                        total_value_depth,
                                        output_depth,
                                        num_heads,
                                        dropout_rate,
                                        batch_size=None,
                                        beam_size=None,
                                        cache=None,
                                        summaries=False,
                                        image_shapes=None,
                                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """

    with tf.variable_scope(
                name,
                default_name="sb_multihead_attention",
                values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
            # self attention
            combined = conv1d(
                query_antecedent,
                total_key_depth * 2 + total_value_depth,
                1,
                name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
        else:
            q = conv1d(
                query_antecedent, total_key_depth, 1, name="q_transform")
            combined = conv1d(
                memory_antecedent,
                total_key_depth + total_value_depth,
                1,
                name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        if memory_antecedent is None:  # decoder self attention (synchronous bidirectional att)
            x = sb_dot_product_attention_for_decoding(q, k, v, bias, batch_size, beam_size, dropout_rate, summaries,
                                                      image_shapes)  # q: [batch, num_heads, length_tmp, lenght]
        else:  # enc-dec attention
            x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = combine_heads(x)
        x = conv1d(x, output_depth, 1, name="output_transform")
        return x


# In[24]:


"""Hyperparameters and ranges common to multiple models."""

def transformer_params():
    """A set of basic hyperparameters."""
    return tf.contrib.training.HParams(
        batching_mantissa_bits=3,
        kernel_height=3,
        kernel_width=1,
        compress_steps=0,
        dropout=0.0,
        clip_grad_norm=0.0,
        initializer="uniform_unit_scaling",
        initializer_gain=1.0,
        label_smoothing=0.1,
        optimizer="Adam",
        optimizer_adam_epsilon=1e-9,
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.998,
        optimizer_momentum_momentum=0.9,
        weight_decay=0.0,
        weight_noise=0.0,
        learning_rate_decay_scheme="noam",
        learning_rate_warmup_steps=16000,
        learning_rate=0.1,
        sampling_method="argmax",  # "argmax" or "random"
        multiply_embedding_mode="sqrt_depth",
        symbol_modality_num_shards=16,
        num_sampled_classes=0,
        shared_source_embedding_and_softmax_weights=int(True),
        shared_target_embedding_and_softmax_weights=int(True),
        pos="timing",
        ffn_layer="conv_hidden_relu",
        attention_key_channels=0,
        attention_value_channels=0,
      
        hidden_size=256,
        batch_size=4096,
        max_length=256,
        filter_size=1024,
        num_heads=4,
        attention_dropout=0.0,
        relu_dropout=0.0,
        residual_dropout=0.1,
        nbr_decoder_problems=1,
        num_hidden_layers=6,
        num_hidden_layers_src=6,
        num_hidden_layers_tgt=6,
      
        # problem hparams
        loss_multiplier=1.4,
        batch_size_multiplier=1,
        max_expected_batch_size_per_shard=64,
        input_modality=None,
        target_modality=None,
        vocab_src_size=37002,
        vocab_tgt_size=37002,
        vocabulary={
        },
    )


def transformer_params_big(data_dir, vocab_src_name, vocab_tgt_name):
    """A set of basic hyperparameters."""
    hparams = transformer_params()
    hparams.vocabulary = {
        "inputs": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_src_name)),
        "targets": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_tgt_name))}
    hparams.hidden_size = 1024
    hparams.filter_size = 4096
    hparams.num_heads = 16
    hparams.batching_mantissa_bits = 3
    return hparams


def transformer_params_base(data_dir, vocab_src_name, vocab_tgt_name):
    """A set of basic hyperparameters."""
    hparams = transformer_params()
    hparams.vocabulary = {
        "inputs": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_src_name)),
        "targets": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_tgt_name))}
    hparams.hidden_size = 256
#     hparams.filter_size = 2048
    hparams.filter_size = 1024
    hparams.num_heads = 8
    hparams.batching_mantissa_bits = 2    
    
    # batch_size=16
    batch_size=16
    
    return hparams


def transformer_params_small(data_dir, vocab_src_name, vocab_tgt_name):
    """A set of basic hyperparameters."""
    hparams = transformer_params()
    hparams.vocabulary = {
        "inputs": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_src_name)),
        "targets": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_tgt_name))}
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.num_heads = 8
    hparams.batching_mantissa_bits = 2
        
    return hparams

def transformer_params_listra(data_dir, vocab_src_name, vocab_tgt_name):
    """A set of custom hyperparameters for LiSTra."""
    hparams = transformer_params()
    hparams.vocabulary = {
        "inputs": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_src_name)),
        "targets": TokenTextEncoder(vocab_filename=os.path.join(data_dir, vocab_tgt_name))}
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.num_heads = 8
    hparams.batching_mantissa_bits = 2
    
    learning_rate=0.1
    max_expected_batch_size_per_shard=64
    batch_size=512
    
    return hparams


# In[25]:


"""Layers common to multiple models."""

# This is a global setting. When turned off, no @function.Defun is used.
allow_defun = True

def flatten4d3d(x):
    """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
    xshape = tf.shape(x)
    result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
    # Preserve static shapes when available.
    xshape_static = x.get_shape()
    result.set_shape([xshape_static[0], None, xshape_static[3]])
    return result


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
    with tf.variable_scope(
            name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        # On the backwards pass, we want to convert the gradient from
        # an indexed-slices to a regular tensor before sending it back to the
        # parameter server. This avoids excess computation on the parameter server.
        embedding_var = ConvertGradientToTensor(embedding_var)
        emb_x = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            emb_x *= multiplier
        shape, static_shape = tf.shape(emb_x), emb_x.shape.as_list()
        if not static_shape or len(static_shape) < 5:
            return emb_x
        # If we had extra channel dimensions, assume it's 1, i.e. shape[3] == 1.
        assert len(static_shape) == 5
        return tf.reshape(emb_x, [shape[0], shape[1], shape[2], static_shape[4]])


def shift_left(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
    return shifted_targets


def shift_left_3d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
    return shifted_targets


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
    """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
        raise ValueError("Inputs to conv must have statically known rank 4.")
    # Add support for left padding.
    if "padding" in kwargs and kwargs["padding"] == "LEFT":
        dilation_rate = (1, 1)
        if "dilation_rate" in kwargs:
            dilation_rate = kwargs["dilation_rate"]
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
        cond_padding = tf.cond(
                tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
                lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
        inputs = tf.pad(inputs, padding)
        # Set middle two dimensions to None to prevent convolution from complaining
        inputs.set_shape([static_shape[0], None, None, static_shape[3]])
        kwargs["padding"] = "VALID"

    def conv2d_kernel(kernel_size_arg, name_suffix):
        """Call conv2d but add suffix to name."""
        if "name" in kwargs:
            original_name = kwargs["name"]
            name = kwargs.pop("name") + "_" + name_suffix
        else:
            original_name = None
            name = "conv_" + name_suffix
        original_force2d = None
        if "force2d" in kwargs:
            original_force2d = kwargs.pop("force2d")
        result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
        if original_name is not None:
            kwargs["name"] = original_name  # Restore for other calls.
        if original_force2d is not None:
            kwargs["force2d"] = original_force2d
        return result

    return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, **kwargs):
    return conv_internal(tf.layers.conv2d, inputs, filters, kernel_size, **kwargs)


def conv1d(inputs, filters, kernel_size, **kwargs):
    return tf.squeeze(
            conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1), **kwargs), 2)
def sb_conv1d(inputs, filters, kernel_size, **kwargs):
  return conv(inputs, filters, (kernel_size, 1), **kwargs)


def separable_conv(inputs, filters, kernel_size, **kwargs):
    return conv_internal(tf.layers.separable_conv2d, inputs, filters, kernel_size, **kwargs)

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


@function.Defun(compiled=True)
def layer_norm_compute_grad(x, epsilon, scale, bias, dy):
    y = layer_norm_compute_python(x, epsilon, scale, bias)
    dx = tf.gradients(ys=[y], xs=[x, epsilon, scale, bias], grad_ys=[dy])
    return dx


@function.Defun(
    compiled=True,
    separate_compiled_gradients=True,
    grad_func=layer_norm_compute_grad)
def layer_norm_compute(x, epsilon, scale, bias):
    return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(
            name, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
                "layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
                "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
        if allow_defun:
            result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
            result.set_shape(x.get_shape())
        else:
            result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def residual_function(hparams):
    """Returns a function for combining layer input and layer output.
    """

    def residual_fn(x, y):
        return hparams.norm_function(x + tf.nn.dropout(
            y, 1.0 - hparams.residual_dropout))

    return residual_fn

def relu_density_logit(x, reduce_dims):
    """logit(density(x)).
    """
    frac = tf.reduce_mean(tf.to_float(x > 0.0), reduce_dims)
    scaled = tf.log(frac + math.exp(-10)) - tf.log((1.0 - frac) + math.exp(-10))
    return scaled


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     summaries=True,
                     dropout=0.0,
                     **kwargs):
    """Hidden layer with RELU activation followed by linear projection."""
    name = kwargs.pop("name") if "name" in kwargs else None
    with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
        if inputs.get_shape().ndims == 3:
            is_3d = True
            inputs = tf.expand_dims(inputs, 2)
        else:
            is_3d = False
        conv_f1 = conv if kernel_size == (1, 1) else separable_conv
        h = conv_f1(
                inputs,
                hidden_size,
                kernel_size,
                activation=tf.nn.relu,
                name="conv1",
                **kwargs)
        if dropout != 0.0:
            h = tf.nn.dropout(h, 1.0 - dropout)
        conv_f2 = conv if second_kernel_size == (1, 1) else separable_conv
        ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
        if is_3d:
            ret = tf.squeeze(ret, 2)
        return ret

def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
    """Pad tensors x and y on axis 1 so that they have the same length."""
    if axis not in [1, 2]:
        raise ValueError("Only axis=1 and axis=2 supported for now.")
    with tf.name_scope("pad_to_same_length", values=[x, y]):
        x_length = tf.shape(x)[axis]
        y_length = tf.shape(y)[axis]
        max_length = tf.maximum(x_length, y_length)
        if final_length_divisible_by > 1:
            # Find the nearest larger-or-equal integer divisible by given number.
            max_length += final_length_divisible_by - 1
            max_length //= final_length_divisible_by
            max_length *= final_length_divisible_by
        length_diff1 = max_length - x_length
        length_diff2 = max_length - y_length

        def padding_list(length_diff, arg):
            if axis == 1:
                return [[[0, 0], [0, length_diff]],
                        tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
            return [[[0, 0], [0, 0], [0, length_diff]],
                    tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

        paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
        paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
        res_x = tf.pad(x, paddings1)
        res_y = tf.pad(y, paddings2)
        # Static shapes are the same except for axis=1.
        x_shape = x.shape.as_list()
        x_shape[axis] = None
        res_x.set_shape(x_shape)
        y_shape = y.shape.as_list()
        y_shape[axis] = None
        res_y.set_shape(y_shape)
        return res_x, res_y


def pad_with_zeros(logits, labels):
    """Pad labels on the length dimension to match logits length."""
    with tf.name_scope("pad_with_zeros", values=[logits, labels]):
        logits, labels = pad_to_same_length(logits, labels)
        if len(labels.shape.as_list()) == 3:  # 2-d labels.
            logits, labels = pad_to_same_length(logits, labels, axis=2)
        return logits, labels


def weights_nonzero(labels):
    """Assign weight 1.0 to all labels except for padding (id=0)."""
    return tf.to_float(tf.not_equal(labels, 0))


def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True):
    """Compute cross-entropy assuming 0s are padding.

    Computes a loss numerator (the sum of losses), and loss denominator
    (the number of non-padding tokens).

    Args:
        logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
        labels: an integer `Tensor` with shape `[batch, timesteps]`.
        label_smoothing: a floating point `Scalar`.
        weights_fn: A function from labels to weights.
        reduce_sum: a Boolean, whether to sum at the end or not.

    Returns:
        loss_numerator: a `Scalar`.  Sum of losses.
        loss_denominator: a `Scalar.  The number of non-padding target tokens.
    """
    confidence = 1.0 - label_smoothing
    vocab_size = tf.shape(logits)[-1]
    with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
        pad_logits, pad_labels = pad_with_zeros(logits, labels)
        xent = smoothing_cross_entropy(pad_logits, pad_labels, vocab_size, confidence)
        weights = weights_fn(pad_labels)
        if not reduce_sum:
            return xent * weights, weights
        return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def smoothing_cross_entropy(logits, labels, vocab_size, confidence):
    """Cross entropy with label smoothing to limit over-confidence."""
    with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
        # Low confidence is given to all non-true labels, uniformly.
        low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
        # Normalizing constant is the best cross-entropy value with soft targets.
        # We subtract it just for readability, makes no difference on learning.
        normalizing = -(confidence * tf.log(confidence) + tf.to_float(
                vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
        # Soft targets.
        soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
        xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets)
        return xentropy - normalizing

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


# In[26]:


"""
Text modality
bottom (embedding): source input, target input
top (cross_entropy_loss): target output
"""

class SymbolModality(object):
    """Modality for sets of discrete symbols.
    Input: Embedding.
    Output: Linear transformation + softmax.
    """

    def __init__(self, model_hparams, vocab_size=None):
        self._model_hparams = model_hparams
        self._vocab_size = vocab_size

    @property
    def name(self):
        return "symbol_modality_%d_%d" % (self._vocab_size, self._body_input_depth)

    @property
    def top_dimensionality(self):
        return self._vocab_size

    @property
    def _body_input_depth(self):
        return self._model_hparams.hidden_size

    def _get_weights(self):
        """Create or get concatenated embedding or softmax variable.
        Returns: a list of self._num_shards Tensors.
        """
        num_shards = self._model_hparams.symbol_modality_num_shards
        shards = []
        for i in range(num_shards):
            shard_size = (self._vocab_size // num_shards) + (
                1 if i < self._vocab_size % num_shards else 0)
            var_name = "weights_%d" % i
            shards.append(
                tf.get_variable(
                    var_name, [shard_size, self._body_input_depth],
                    initializer=tf.random_normal_initializer(
                        0.0, self._body_input_depth ** -0.5)))
        if num_shards == 1:
            ret = shards[0]
        else:
            ret = tf.concat(shards, 0)
        ret = ConvertGradientToTensor(ret)
        return ret

    def bottom_simple(self, x, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            # Squeeze out the channels dimension.
            x = tf.squeeze(x, axis=3)
            var = self._get_weights()
            ret = tf.gather(var, x)
            if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
                ret *= self._body_input_depth ** 0.5
            ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
            return ret

    def bottom(self, x):
        if self._model_hparams.shared_source_embedding_and_softmax_weights:
            return self.bottom_simple(x, "shared", reuse=None)
        else:
            return self.bottom_simple(x, "input_emb", reuse=None)

    def targets_bottom(self, x):
        if self._model_hparams.shared_target_embedding_and_softmax_weights:
            return self.bottom_simple(x, "shared", reuse=tf.AUTO_REUSE)
        else:
            return self.bottom_simple(x, "target_emb", reuse=tf.AUTO_REUSE)

    def top(self, body_output, targets):
        """Generate logits.
        Args:
            body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
            targets: A Tensor with shape [batch, p0, p1, 1]
        Returns:
            logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
        """
        if self._model_hparams.shared_target_embedding_and_softmax_weights:
            scope_name = "shared"
            reuse = True
        else:
            scope_name = "softmax"
            # reuse = False
            reuse = tf.AUTO_REUSE
        with tf.variable_scope(scope_name, reuse=reuse):
            var = self._get_weights()
            shape = tf.shape(body_output)[:-1]
            body_output = tf.reshape(body_output, [-1, self._body_input_depth])
            logits = tf.matmul(body_output, var, transpose_b=True)
            logits = tf.reshape(logits, tf.concat([shape, [self._vocab_size]], 0))
            # insert a channels dimension
            return tf.expand_dims(logits, 3)

    def bottom_sharded(self, xs, data_parallelism):
        """Transform the inputs.
            [batch, p0, p1, depth --> [batch, p0, p1, body_input_depth].
        """
        return data_parallelism(self.bottom, xs)

    def targets_bottom_sharded(self, xs, data_parallelism):
        """Transform the targets.
            [batch, p0, p1, target_channels] --> [batch, p0, p1, body_input_depth].
        """
        return data_parallelism(self.targets_bottom, xs)

    def top_sharded(self,
                    sharded_body_output,
                    sharded_targets,
                    data_parallelism,
                    weights_fn=weights_nonzero):
        """Transform all shards of targets.
        Classes with cross-shard interaction will override this function.
        """
        sharded_logits = data_parallelism(self.top, sharded_body_output,
                                          sharded_targets)
        if sharded_targets is None:
            return sharded_logits, 0

        loss_num, loss_den = data_parallelism(
            padded_cross_entropy,
            sharded_logits,
            sharded_targets,
            self._model_hparams.label_smoothing,
            weights_fn=weights_fn)
        loss = tf.add_n(loss_num) / tf.maximum(1.0, tf.add_n(loss_den))
        return sharded_logits, loss


# In[27]:


"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""



class Transformer(object):
    """Attention net.  See file docstring."""

    def __init__(self,
                 hparams,
                 mode,
                 data_parallelism=None):
    
        hparams = copy.copy(hparams)
        hparams.add_hparam("mode", mode)
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            for key in hparams.values():
                if key[-len("dropout"):] == "dropout":
                    setattr(hparams, key, 0.0)
        self._hparams = hparams
        self._data_parallelism = data_parallelism
        self._num_datashards = data_parallelism.n   
        # source side
        self._hparams.input_modality = SymbolModality(hparams, hparams.vocab_src_size)
        # target side
        self._hparams.target_modality = SymbolModality(hparams, hparams.vocab_tgt_size)
        
    def infer(self,
              features=None,
              decode_length=50,
              beam_size=1,
              top_beams=1,
              alpha=0.0):
        """
        A inference method.
        """
        local_features = dict()
        local_features["_num_datashards"] = self._num_datashards
        local_features["_data_parallelism"] = self._data_parallelism
        local_features["_hparams"] = self._hparams
        local_features["_shard_features"] = self._shard_features
        local_features["encode"] = self.encode
        local_features["decode"] = self.decode

        tf.logging.info("Beam Decoding with beam size %d" % beam_size)
        return inference._beam_decode(features, decode_length, beam_size, top_beams, alpha, local_features)

    def _shard_features(self, features):  # pylint: disable=missing-docstring
        sharded_features = dict()
        for k, v in six.iteritems(features):
            v = tf.convert_to_tensor(v)
            if not v.shape.as_list():
                v = tf.expand_dims(v, axis=-1)
                v = tf.tile(v, [self._num_datashards])
            sharded_features[k] = self._data_parallelism(tf.identity, tf.split(v, self._num_datashards, 0))

        return sharded_features

    def model_fn(self, features):
        """Computes the entire model and produces sharded logits and training loss.
        """
    
        start_time = time.time()
        dp = self._data_parallelism
        sharded_features = self._shard_features(features)    
        transformed_features = {}
    
        # source embedding
        with tf.variable_scope(self._hparams.input_modality.name, reuse=False):
            transformed_features["inputs"] = sharded_features["inputs"]
    
        # target embedding
        with tf.variable_scope(self._hparams.target_modality.name, reuse=False):
            transformed_features["targets_l1"] = self._hparams.target_modality.targets_bottom_sharded(
                    sharded_features["targets_l1"], dp)
            transformed_features["targets_l2"] = self._hparams.target_modality.targets_bottom_sharded(
                    sharded_features["targets_l2"], dp)

        # Construct the model body.
        with tf.variable_scope("body", reuse=False):
            with tf.name_scope("model"):
                datashard_to_features = [{
                    k: v[d] for k, v in six.iteritems(transformed_features)
                    } for d in range(self._num_datashards)]
                body_outputs = self._data_parallelism(self.model_fn_body, datashard_to_features)
                extra_loss = 0.
    
        body_outputs_l1 = []
        body_outputs_l2 = []
        # for multi-gpus
        for output in body_outputs:
            body_outputs_l1.append(output[0])
            body_outputs_l2.append(output[1])
    
        # target linear transformation and compute loss
        with tf.variable_scope(self._hparams.target_modality.name, reuse=False):  ## = target_reuse
            sharded_logits, training_loss_l1 = (self._hparams.target_modality.top_sharded(
                body_outputs_l1, sharded_features["targets_l1"], self._data_parallelism))
            sharded_logits_l2, training_loss_l2 = (self._hparams.target_modality.top_sharded(
                body_outputs_l2, sharded_features["targets_l2"], self._data_parallelism))
            # TODO: this heps to choise if we need to combine the lossesh
            training_loss = training_loss_l1 + training_loss_l2
            training_loss *= self._hparams.loss_multiplier

#             training_loss = training_loss_l1
#             training_loss *= 1.0

        tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
        return sharded_logits, training_loss, extra_loss

    def model_fn_body(self, features):
        hparams = copy.copy(self._hparams)
        inputs = features.get("inputs")

        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, hparams)

        targets_l1 = features["targets_l1"]
        targets_l2 = features["targets_l2"]
        targets_l1 = flatten4d3d(targets_l1)
        targets_l2 = flatten4d3d(targets_l2)
        (decoder_input, decoder_self_attention_bias) = transformer_prepare_decoder(
                targets_l1, targets_l2, hparams)

        decode_output = self.decode(decoder_input, encoder_output, encoder_decoder_attention_bias,
                decoder_self_attention_bias, hparams)

        return decode_output

    def encode(self, inputs, hparams):
        # inputs is audio feature in 3d [batch, length, raw_features]
        (encoder_input, self_attention_bias, encoder_decoder_attention_bias) =             transformer_prepare_encoder(inputs, hparams)

        encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
        encoder_output = transformer_encoder(encoder_input, self_attention_bias, hparams)

        return encoder_output, encoder_decoder_attention_bias

    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias,
               decoder_self_attention_bias, hparams, batch_size=None, beam_size=None, cache=None):
        decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.residual_dropout)
      
        if cache is None:  # training
            decoder_output = transformer_decoder(
                    decoder_input, encoder_output, decoder_self_attention_bias,
                    encoder_decoder_attention_bias, hparams, cache=cache)
            return tf.expand_dims(decoder_output, axis=3)
        else:  # inference
            decoder_output = transformer_decoder_for_decoding(
                    decoder_input, encoder_output, decoder_self_attention_bias,
                    encoder_decoder_attention_bias, hparams, batch_size, beam_size, cache=cache)
            return tf.expand_dims(decoder_output, axis=2)


def transformer_prepare_encoder(inputs, hparams):
    """Prepare one shard of the model for the encoder.
    """
    # Flatten inputs.
    ishape_static = inputs.shape.as_list()
    encoder_input = inputs
    encoder_padding = embedding_to_padding(encoder_input)
    ignore_padding = attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding

    if hparams.pos == "timing":
        encoder_input = add_timing_signal_1d(encoder_input)
    return encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias


def transformer_prepare_decoder(targets_l1, targets_l2, hparams):
    """Prepare one shard of the model for the decoder.
    """
    decoder_self_attention_bias = (
        attention_bias_lower_triangle(tf.shape(targets_l1)[1])) ## [1, 1, length, length]
    decoder_input_l1 = shift_left_3d(targets_l1)
    decoder_input_l2 = shift_left_3d(targets_l2)
    if hparams.pos == "timing":
        decoder_input_l1 = add_timing_signal_1d(decoder_input_l1)
        decoder_input_l2 = add_timing_signal_1d(decoder_input_l2)
    decoder_input = tf.concat([tf.expand_dims(decoder_input_l1, 0), tf.expand_dims(decoder_input_l2, 0)], axis=0)
    # [2, batch, length, hidden_size]
    return decoder_input, decoder_self_attention_bias


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        use_fc=True):
    """A stack of transformer layers.
    """
    x = encoder_input
    # Summaries don't work in multi-problem setting yet.
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        if use_fc:
            x = tf.layers.dense(inputs=x,
                                units=hparams.hidden_size,
                                activation=None,
                                use_bias=False,
                                name='full_connect')
        for layer in range(hparams.num_hidden_layers_src):
            with tf.variable_scope("layer_%d" % layer):
                y = multihead_attention(
                    x,
                    None,
                    encoder_self_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    summaries=summaries,
                    name="encoder_self_attention")
                x = residual_fn(x, y, hparams) ###
                y = transformer_ffn_layer(x, hparams)
                x = residual_fn(x, y, hparams)
    return x


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder"):
    """A stack of transformer layers.
    """
    x = decoder_input
    # Summaries don't work in multi-problem setting yet.
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        for layer in range(hparams.num_hidden_layers_tgt):
            layer_name = "layer_%d" % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                y = sb_multihead_attention(
                    x,
                    None,
                    decoder_self_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    cache=layer_cache,
                    summaries=summaries,
                    name="decoder_self_attention")
                x = residual_fn(x, y, hparams)
                y = sb_multihead_attention(
                    x,
                    encoder_output,
                    encoder_decoder_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    summaries=summaries,
                    name="encdec_attention")
                x = residual_fn(x, y, hparams)
                y = transformer_ffn_layer(x, hparams)
                x = residual_fn(x, y, hparams)
    return x


def transformer_decoder_for_decoding(decoder_input,
                                     encoder_output,
                                     decoder_self_attention_bias,
                                     encoder_decoder_attention_bias,
                                     hparams,
                                     batch_size=None,
                                     beam_size=None,
                                     cache=None,
                                     name="decoder"):
    """A stack of transformer layers.
    """
    x = decoder_input
    # Summaries don't work in multi-problem setting yet.
    summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
    with tf.variable_scope(name):
        for layer in range(hparams.num_hidden_layers_tgt):
            layer_name = "layer_%d" % layer
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                y = sb_multihead_attention_for_decoding(
                        x,
                        None,
                        decoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        batch_size,
                        beam_size,
                        cache=layer_cache,
                        summaries=summaries,
                        name="decoder_self_attention")
                x = residual_fn(x, y, hparams)
                y = sb_multihead_attention_for_decoding(
                        x,
                        encoder_output,
                        encoder_decoder_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        summaries=summaries,
                        name="encdec_attention")
                x = residual_fn(x, y, hparams)
                y = transformer_ffn_layer(x, hparams)
                x = residual_fn(x, y, hparams)
    return x


def transformer_ffn_layer(x, hparams):
    """Feed-forward layer in the transformer.
    [batch_size, length, hparams.hidden_size] -->  [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == "conv_hidden_relu":
        return conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            dropout=hparams.relu_dropout)
    else:
        assert hparams.ffn_layer == "none"
        return x


# # Train

# In[28]:


flags = tf.flags
FLAGS = flags.FLAGS
# TODO : I need to make this sync with the run hparam
flags.DEFINE_string("pretrain_output_dir", "../../../../data/processed/pretrain_model", "Base output directory for run.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("data_dir", "../../../../data/processed/tf_data", "Directory with training data.")
flags.DEFINE_string("train_src_name", "2m.bpe.unk.zh", "src name of training data.")
flags.DEFINE_string("train_tgt_name", "2m.bpe.unk.en", "tgt name of training data.")
flags.DEFINE_string("vocab_src_name", "en.vocab", "src name of vocab.")
flags.DEFINE_string("vocab_tgt_name", "ln.vocab", "tgt name of vocab.")
flags.DEFINE_integer("vocab_src_size", 30000, "source vocab size.")
flags.DEFINE_integer("vocab_tgt_size", 30000, "target vocab size.")

# Model
flags.DEFINE_string("model", "Transformer", "Which model to use.")
flags.DEFINE_string("hparams_set", "transformer_params_base", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string("hparams", "", """A comma-separated list of `name=value` hyperparameter values.""")
flags.DEFINE_integer("train_steps", 80000, "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_bool("eval_print", False, "Print eval logits and predictions.")
flags.DEFINE_integer("keep_checkpoint_max", 20, "How many recent checkpoints to keep.")
flags.DEFINE_integer("save_checkpoint_secs", 0, "How seconds to save checkpoints.")
flags.DEFINE_integer("save_checkpoint_steps", 1000, "How steps tp save checkpoints.")
flags.DEFINE_float("gpu_mem_fraction", 0.95, "How GPU memory to use.")
flags.DEFINE_bool("experimental_optimize_placement", False,
                  "Optimize ops placement with experimental session options.")

# Distributed training flags
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "local_run",
                    "Method of tf.contrib.learn.Experiment to run.")
flags.DEFINE_bool("locally_shard_to_cpu", False,
                  "Use CPU as a sharding device runnning locally. This allows "
                  "to test sharded model construction on a machine with 1 GPU.")
flags.DEFINE_bool("daisy_chain_variables", True,
                  "copy variables around in a daisy chain")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining gpus."
                    " e.g. \"1 3 2 4\"")

# Decode flags
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file_l1", None, "Path to inference output file")
flags.DEFINE_string("decode_to_file_l2", None, "Path to inference output file")
flags.DEFINE_integer("decode_extra_length", 100, "Added decode length.")
flags.DEFINE_integer("decode_batch_size", 32, "Batch size for decoding. ")
flags.DEFINE_integer("decode_beam_size", 4, "The beam size for beam decoding")
flags.DEFINE_float("decode_alpha", 0.6, "Alpha for length penalty")
flags.DEFINE_bool("decode_return_beams", False,"whether return all beams or one")

# Audio configuration
flags.DEFINE_integer("dim_feature", 80, "Batch size for decoding. ")
flags.DEFINE_integer("num_context", 2, "Batch size for decoding. ")
flags.DEFINE_integer("downsample", 3, "Batch size for decoding. ")

flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen", "Temporary storage directory.")
flags.DEFINE_integer("num_shards", 10, "How many shards to use.")
flags.DEFINE_integer("max_cases", 0, "Maximum number of cases to generate (unbounded if 0).")
flags.DEFINE_integer("random_seed", 429459, "Random seed to use.")


# In[29]:


# tf.flags.FLAGS.__flags 


# In[30]:


# # from utils import generator_utils
# # from utils import trainer_utils as trainer_utils

# flags = tf.flags
# FLAGS = flags.FLAGS


UNSHUFFLED_SUFFIX = "-unshuffled"

_SUPPORTED_PROBLEM_GENERATORS = {
    "translation": (
        lambda: generator_utils.translation_token_generator(FLAGS.data_dir, FLAGS.tmp_dir, 
            FLAGS.train_src_name, FLAGS.train_tgt_name, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) )
}


def set_random_seed():
    """Set the random seed from flag everywhere."""
    tf.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)


def generate_data():
    data_dir = os.path.expanduser(FLAGS.data_dir)
    tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
    tf.gfile.MakeDirs(data_dir)
    tf.gfile.MakeDirs(tmp_dir)
  
    problem = list(sorted(_SUPPORTED_PROBLEM_GENERATORS))[0]
    set_random_seed()

    training_gen = _SUPPORTED_PROBLEM_GENERATORS[problem]

    tf.logging.info("Generating training data for %s.", problem)
    train_output_files = generator_utils.generate_files(
            training_gen(), problem + UNSHUFFLED_SUFFIX + "-train",
            FLAGS.data_dir, FLAGS.num_shards, FLAGS.max_cases)

    train_output_files = []
    output_dir = FLAGS.data_dir
    for shard in range(FLAGS.num_shards):
        output_filename = "%s-%.5d-of-%.5d" % ('translation-unshuffled-train', shard, FLAGS.num_shards)
        output_file = os.path.join(output_dir, output_filename)
        train_output_files.append(output_file)

    tf.logging.info("Shuffling data...")
    for fname in train_output_files:
        records = generator_utils.read_records(fname)
        random.shuffle(records)
        out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
        generator_utils.write_records(records, out_fname)
        tf.gfile.Remove(fname)
    tf.logging.info("Data Process Over")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.generate_data:
        generate_data()
        if FLAGS.model != "transformer":
            return
  
    run(model=FLAGS.model,
        output_dir=FLAGS.output_dir)


# In[31]:


import sys
sys.argv=['']; del sys 

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_dir", default='./', help="Temporary storage directory.")
    parser.add_argument("--data_dir", default='./', help="Directory with training data.")
    
    parser.add_argument("--train_csv_name", default='./', help="Filename of training data.")
    parser.add_argument("--dev_csv_name", default='./', help="Filename of dev data.")
    parser.add_argument("--test_csv_name", default='./', help="Filename of test data.")
    
    parser.add_argument("--wav_dir_train", default='./', help="Wavefile path of training data.")
    parser.add_argument("--wav_dir_dev", default='./', help="Wavefile path of dev data.")
    parser.add_argument("--wav_dir_test", default='./', help="Wavefile path of test data.")
    
    parser.add_argument("--vocabA_name", default='./', help="Vocab language A file name.")
    parser.add_argument("--vocabB_name", default='./', help="Vocab language B file name.")
    
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size.")
    
    parser.add_argument("-d", "--dim_raw_input", type=int, default=80, help="The dimension of input feature.")
    args = parser.parse_args()
    return args

args = get_argument()


# In[32]:


args.tmp_dir = "../../../../data/processed"
args.data_dir = "../../../../data/processed/tf_data"

args.train_csv_name = "tf_data/train.en-ln.csv"
args.dev_csv_name = "tf_data/test.en-ln.csv"
args.test_csv_name = "tf_data/test.en-ln.csv"

args.wav_dir_train = "../../../../../LiSTra/dataset/english/wav_verse"
args.wav_dir_dev = "../../../../../LiSTra/dataset/english/wav_verse"
args.wav_dir_test = "../../../../../LiSTra/dataset/english/wav_verse"

args.vocabA_name = "tf_data/en.vocab"
args.vocabB_name = "tf_data/ln.vocab"
args.vocab_size = 30000
args.dim_raw_input = 80

args.gpu_mem_fraction = 0.95

args.worker_gpu = 1
args.vocab_src_size = 30000
args.vocab_tgt_size = 30000

args.vocab_src_name = "en.vocab"
args.vocab_src_name = "ln.vocab"

args.hparams_set = "transformer_params_base"
args.train_steps = 80000
args.keep_checkpoint_max = 50

args.output_dir = "../../../../data/processed"
args.pretrain_output_dir = "../../../../data/processed/pretrain_model"


# In[ ]:


tf.app.run(main=main)


# In[ ]:


# get_ipython().system(' ls ../../../../data/processed/pretrain_model/')


# In[ ]:


# tf.app.run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)


# In[ ]:


# get_ipython().system('pip install tensorflow==1.13.0rc1')
# get_ipython().system('pip install ipdb')


# In[ ]:


# export CUDA_VISIBLE_DEVICES=0 #1, 2, 3

# python3 ./run.py      --worker_gpu=1     --gpu_mem_fraction=0.95     --data_dir=../../../../data/processed/tf_data     --vocab_src_size=30000      --vocab_tgt_size=30000      --vocab_src_name=en.vocab     --vocab_tgt_name=ln.vocab     --hparams_set=transformer_params_base    --train_steps=80000      --keep_checkpoint_max=50      --output_dir=../../../../data/processed
#         --pretrain_output_dir=../../../../data/processed/pretrain_model

#     --train_steps=20000  \


# export CUDA_VISIBLE_DEVICES=0

# python ./run.py  \
#     --worker_gpu=2 \
#     --gpu_mem_fraction=0.95 \
#     --data_dir=../../../../data/external/TED_Speech_Translation/tf_data \
#     --vocab_src_size=30000  \
#     --vocab_tgt_size=30000  \
#     --vocab_src_name=en.vocab \
#     --vocab_tgt_name=fr.vocab \
#     --hparams_set=transformer_params_base  \
#     --train_steps=50100  \
#     --keep_checkpoint_max=50  \
#     --output_dir=../../../../data/external/TED_Speech_Translation/train_models_enfr \
#     --pretrain_output_dir=../../../../data/external/TED_Speech_Translation/train_models_enfr/pretrain_model 


# # Test

# In[ ]:


# # export CUDA_VISIBLE_DEVICES=0


# python run.py     --gpu_mem_fraction=0.8     --hparams=''     --data_dir=../../../../data/external/LiSTra/tf_data     --hparams_set=transformer_params_base     --output_dir=../../../../data/external/LiSTra/train_models_enln     --vocab_src_size=30000     --vocab_tgt_size=30000     --vocab_src_name=en.vocab    --vocab_tgt_name=ln.vocab    --train_steps=0     --decode_beam_size=8     --decode_alpha=1.0     --decode_batch_size=16      --decode_from_file=./     --decode_to_file_l1=test.en.output     --decode_to_file_l2=test.ln.output     --decode_return_beams=False


# # python run.py \
# #     --gpu_mem_fraction=0.8 \
# #     --hparams='' \
# #     --data_dir=../../../../data/external/TED_Speech_Translation/tf_data \
# #     --hparams_set=transformer_params_base \
# #     --output_dir=../../../../data/external/TED_Speech_Translation/train_models_enfr \
# #     --vocab_src_size=30000 \
# #     --vocab_tgt_size=30000 \
# #     --vocab_src_name=en-fr.vocab \
# #     --vocab_tgt_name=en-fr.vocab \
# #     --train_steps=0 \
# #     --decode_beam_size=8 \
# #     --decode_alpha=1.0 \
# #     --decode_batch_size=16  \
# #     --decode_from_file=./ \
# #     --decode_to_file_l1=tst2015.en.output \
# #     --decode_to_file_l2=tst2015.fr.output \
# #     --decode_return_beams=False


# # In[ ]:


# get_ipython().system('cat train.sh')


# # In[ ]:





# # # Evaluate

# # In[ ]:


# hyp=$1
# ref=$hyp.ref
# ../../../../data/external/LiSTra/train_models_enln/rapport/wait-2-0.28/test.ln.output.ref
# sed -r 's/(@@ )|(@@ ?$)//g' $hyp > $hyp.out
# sed -r 's/(@@ )|(@@ ?$)//g' $ref > $ref.out
# # remove the delay symbol
# sed -i 's/.\{30\}//' $hyp.out
# sed -i 's/.\{30\}//' $ref.out
# #perl chi_char_segment.pl -t plain < $hyp.out > $hyp.seg
# #perl chi_char_segment.pl -t plain < $ref.out > $ref.seg
# #mv $hyp.seg $hyp.out
# #mv $ref.seg $ref.out
# perl multi-bleu.perl $ref.out < $hyp.out
# rm $hyp.out
# rm $ref.out

# # bash blue.sh ../../../../data/external/LiSTra/train_models_enln/rapport/wait-2-0.28/test.en.output

# # bash blue.sh ../../../../data/external/LiSTra/train_models_enln/rap
# # port/wait-2-0.28/test.ln.output


# # In[ ]:


# import sys
# import os
# import numpy


# def editDistance(r, h):
#     '''
#     This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
#     Main algorithm used is dynamic programming.
#     Attributes:
#         r -> the list of words produced by splitting reference sentence.
#         h -> the list of words produced by splitting hypothesis sentence.
#     '''
#     d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
#     for i in range(len(r) + 1):
#         for j in range(len(h) + 1):
#             if i == 0:
#                 d[0][j] = j
#             elif j == 0:
#                 d[i][0] = i
#     for i in range(1, len(r) + 1):
#         for j in range(1, len(h) + 1):
#             if r[i - 1] == h[j - 1]:
#                 d[i][j] = d[i - 1][j - 1]
#             else:
#                 substitute = d[i - 1][j - 1] + 1
#                 insert = d[i][j - 1] + 1
#                 delete = d[i - 1][j] + 1
#                 d[i][j] = min(substitute, insert, delete)
#     return d


# def getStepList(r, h, d):
#     '''
#     This function is to get the list of steps in the process of dynamic programming.
#     Attributes:
#         r -> the list of words produced by splitting reference sentence.
#         h -> the list of words produced by splitting hypothesis sentence.
#         d -> the matrix built when calulating the editting distance of h and r.
#     '''
#     x = len(r)
#     y = len(h)
#     wer_list = []
#     match_list = []
#     while True:
#         if x == 0 and y == 0:
#             break
#         elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
#             wer_list.append("e")
#             x = x - 1
#             y = y - 1
#         elif y >= 1 and d[x][y] == d[x][y - 1] + 1:
#             wer_list.append("i")
#             x = x
#             y = y - 1
#         elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + 1:
#             wer_list.append("s")
#             match_list.append((x - 1, y - 1))
#             x = x - 1
#             y = y - 1
#         else:
#             wer_list.append("d")
#             x = x - 1
#             y = y
#     return wer_list[::-1], match_list


# def alignedPrint(wer_list, r, h, result):
#     '''
#     This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.

#     Attributes:
#         list   -> the list of steps.
#         r      -> the list of words produced by splitting reference sentence.
#         h      -> the list of words produced by splitting hypothesis sentence.
#         result -> the rate calculated based on edit distance.
#     '''
#     ref = []
#     hyp = []
#     error_type = []
#     for i in range(len(wer_list)):
#         if wer_list[i] == "i":
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "d":
#                     count += 1
#             index = i - count
#             ref += " " * (len(h[index])),
#         elif wer_list[i] == "s":
#             count1 = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count1 += 1
#             index1 = i - count1
#             count2 = 0
#             for j in range(i):
#                 if wer_list[j] == "d":
#                     count2 += 1
#             index2 = i - count2
#             if len(r[index1]) < len(h[index2]):
#                 ref += r[index1] + " " * (len(h[index2]) - len(r[index1])),
#             else:
#                 ref += r[index1],
#         else:
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count += 1
#             index = i - count
#             ref += r[index],

#     # print "HYP:",
#     for i in range(len(wer_list)):
#         if wer_list[i] == "d":
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count += 1
#             index = i - count
#             hyp += " " * (len(r[index])),
#         elif wer_list[i] == "s":
#             count1 = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count1 += 1
#             index1 = i - count1
#             count2 = 0
#             for j in range(i):
#                 if wer_list[j] == "d":
#                     count2 += 1
#             index2 = i - count2
#             if len(r[index1]) > len(h[index2]):
#                 hyp += h[index2] + " " * (len(r[index1]) - len(h[index2])),
#             else:
#                 hyp += h[index2],
#         else:
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "d":
#                     count += 1
#             index = i - count
#             hyp += h[index],

#     # print "EVA:",
#     for i in range(len(wer_list)):
#         if wer_list[i] == "d":
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count += 1
#             index = i - count
#             error_type += "D" + " " * (len(r[index]) - 1),
#         elif wer_list[i] == "i":
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "d":
#                     count += 1
#             index = i - count
#             error_type += "I" + " " * (len(h[index]) - 1),
#         elif wer_list[i] == "s":
#             count1 = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count1 += 1
#             index1 = i - count1
#             count2 = 0
#             for j in range(i):
#                 if wer_list[j] == "d":
#                     count2 += 1
#             index2 = i - count2
#             if len(r[index1]) > len(h[index2]):
#                 error_type += "S" + " " * (len(r[index1]) - 1),
#             else:
#                 error_type += "S" + " " * (len(h[index2]) - 1),
#         else:
#             count = 0
#             for j in range(i):
#                 if wer_list[j] == "i":
#                     count += 1
#             index = i - count
#             error_type += " " * (len(r[index])),
#     return ' '.join(ref), ' '.join(hyp), ' '.join(error_type)


# def wer(r, h):
#     """
#     This is a function that calculate the word error rate in ASR.
#     You can use it like this: wer("what is it".split(), "what is".split())
#     """
#     # build the matrix
#     d = editDistance(r, h)
#     # wer_list, match_list = getStepList(r, h, d)
#     # ref, hyp, error_type = alignedPrint(wer_list, r, h, 0)
#     return d[len(r)][len(h)], len(r) #, ref, hyp, error_type


# def main():
#     ref_f = open(sys.argv[1],'r')
#     hyp_f = open(sys.argv[2],'r')
#     index = 0
#     total_error = 0
#     total_ref = 0

#     for line1, line2 in zip(ref_f, hyp_f):
#         line1, line2 = line1.strip(), line2.strip()
      
#         error_num, ref_len = wer(line1.split(), line2.split())
#         # error_num, ref_len = wer(line1, line2)
#         total_error += error_num
#         total_ref += ref_len
#     print('WER is {0}'.format(total_error*1.0/total_ref))


# if __name__ == '__main__':
#     main()


# # In[ ]:


# hyp=$1
# ref=$hyp.ref #../data/raw_data/tst2015.en
# sed -r 's/(@@ )|(@@ ?$)//g' $hyp > $hyp.out
# sed -r 's/(@@ )|(@@ ?$)//g' $ref > $ref.out
# sed -i 's/.\{6\}//' $hyp.out
# sed -i 's/.\{6\}//' $ref.out
# python wer.py $ref.out  $hyp.out
# rm $hyp.out
# rm $ref.out


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:


# get_ipython().system('python -m utils.tfRecord     --tmp_dir=../../../../data/processed     --data_dir=../../../../data/processed/tf_data     --train_csv_name=tf_data/train.en-ln.csv     --dev_csv_name=tf_data/test.en-ln.csv     --test_csv_name=tf_data/test.en-ln.csv     --wav_dir_train=../../../../../LiSTra/dataset/english/wav_verse     --wav_dir_dev=../../../../../LiSTra/dataset/english/wav_verse     --wav_dir_test=../../../../../LiSTra/dataset/english/wav_verse     --vocabA_name=tf_data/en.vocab     --vocabB_name=tf_data/ln.vocab     --vocab_size=30000     --dim_raw_input=80')


# # In[ ]:





# # In[ ]:





# # In[ ]:


# get_ipython().system('pip install matplotlib')


# # In[ ]:




