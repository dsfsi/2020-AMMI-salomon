"""
Text modality
bottom (embedding): source input, target input
top (cross_entropy_loss): target output
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from utils import parallel
from models import common_layers


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
        ret = parallel.ConvertGradientToTensor(ret)
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
                    weights_fn=common_layers.weights_nonzero):
        """Transform all shards of targets.
        Classes with cross-shard interaction will override this function.
        """
        sharded_logits = data_parallelism(self.top, sharded_body_output,
                                          sharded_targets)
        if sharded_targets is None:
            return sharded_logits, 0

        loss_num, loss_den = data_parallelism(
            common_layers.padded_cross_entropy,
            sharded_logits,
            sharded_targets,
            self._model_hparams.label_smoothing,
            weights_fn=weights_fn)
        loss = tf.add_n(loss_num) / tf.maximum(1.0, tf.add_n(loss_den))
        return sharded_logits, loss


