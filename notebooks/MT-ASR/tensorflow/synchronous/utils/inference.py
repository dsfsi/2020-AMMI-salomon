"""Implemetation of beam seach with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models import common_layers
from models import common_attention
from utils.beamsearch import *
import tensorflow as tf


# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7


def _beam_decode(features, decode_length, beam_size, top_beams, alpha, local_features):

    decoded_ids_l1, decoded_ids_l2, scores_l1, scores_l2 = \
        _fast_decode(features, decode_length, beam_size, top_beams, alpha, local_features)
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
        timing_signal = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
        """Performs preprocessing steps on the targets to prepare for the decoder.
        Returns: Processed targets [batch_size, 1, hidden_dim]
        """
        # _shard_features called to ensure that the variable names match
        targets = local_features["_shard_features"]({"targets": targets})["targets"]
        with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
        targets = common_layers.flatten4d3d(targets)

        # TODO(llion): Explain! Is this even needed?
        targets = tf.cond(
            tf.equal(i, 0), lambda: tf.concat([tf.zeros_like(targets)[:,:1,:],targets[:,1:,:]], axis=1), lambda: targets)
        
        if hparams.pos == "timing":
            timing_signal_1 = tf.cond(
                        tf.equal(i, 0), lambda: timing_signal[:, i:i + 2], lambda: timing_signal[:, i+1:i + 2])
            targets += timing_signal_1
        return targets

    decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length+1))

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

