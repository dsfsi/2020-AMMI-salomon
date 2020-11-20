"""Implemetation of beam seach with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.util import nest
from utils.tfRecord import RESERVED_TOKENS_TO_INDEX

# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7
L1_SYMBOL = RESERVED_TOKENS_TO_INDEX["<2L1>"]
L2_SYMBOL = RESERVED_TOKENS_TO_INDEX["<2L2>"]
DELAY_SYMBOL = RESERVED_TOKENS_TO_INDEX["<DELAY>"]


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
