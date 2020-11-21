"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import six
from six.moves import xrange
import tensorflow as tf
import time

from models import common_attention
from models import common_layers
from utils import inference
from models.modality import SymbolModality


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
            training_loss = training_loss_l1 + training_loss_l2
            # training_loss = training_loss_l1
      
            training_loss *= self._hparams.loss_multiplier

        tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
        return sharded_logits, training_loss, extra_loss

    def model_fn_body(self, features):
        hparams = copy.copy(self._hparams)
        inputs = features.get("inputs")

        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs, hparams)

        targets_l1 = features["targets_l1"]
        targets_l2 = features["targets_l2"]
        targets_l1 = common_layers.flatten4d3d(targets_l1)
        targets_l2 = common_layers.flatten4d3d(targets_l2)
        (decoder_input, decoder_self_attention_bias) = transformer_prepare_decoder(
                targets_l1, targets_l2, hparams)

        decode_output = self.decode(decoder_input, encoder_output, encoder_decoder_attention_bias,
                decoder_self_attention_bias, hparams)

        return decode_output

    def encode(self, inputs, hparams):
        # inputs is audio feature in 3d [batch, length, raw_features]
        (encoder_input, self_attention_bias, encoder_decoder_attention_bias) = \
            transformer_prepare_encoder(inputs, hparams)

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
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding

    if hparams.pos == "timing":
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
    return encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias


def transformer_prepare_decoder(targets_l1, targets_l2, hparams):
    """Prepare one shard of the model for the decoder.
    """
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(targets_l1)[1])) ## [1, 1, length, length]
    decoder_input_l1 = common_layers.shift_left_3d(targets_l1)
    decoder_input_l2 = common_layers.shift_left_3d(targets_l2)
    if hparams.pos == "timing":
        decoder_input_l1 = common_attention.add_timing_signal_1d(decoder_input_l1)
        decoder_input_l2 = common_attention.add_timing_signal_1d(decoder_input_l2)
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
                y = common_attention.multihead_attention(
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
                x = common_attention.residual_fn(x, y, hparams) ###
                y = transformer_ffn_layer(x, hparams)
                x = common_attention.residual_fn(x, y, hparams)
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
                y = common_attention.sb_multihead_attention(
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
                x = common_attention.residual_fn(x, y, hparams)
                y = common_attention.sb_multihead_attention(
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
                x = common_attention.residual_fn(x, y, hparams)
                y = transformer_ffn_layer(x, hparams)
                x = common_attention.residual_fn(x, y, hparams)
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
                y = common_attention.sb_multihead_attention_for_decoding(
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
                x = common_attention.residual_fn(x, y, hparams)
                y = common_attention.sb_multihead_attention_for_decoding(
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
                x = common_attention.residual_fn(x, y, hparams)
                y = transformer_ffn_layer(x, hparams)
                x = common_attention.residual_fn(x, y, hparams)
    return x


def transformer_ffn_layer(x, hparams):
    """Feed-forward layer in the transformer.
    [batch_size, length, hparams.hidden_size] -->  [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == "conv_hidden_relu":
        return common_layers.conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            dropout=hparams.relu_dropout)
    else:
        assert hparams.ffn_layer == "none"
        return x
