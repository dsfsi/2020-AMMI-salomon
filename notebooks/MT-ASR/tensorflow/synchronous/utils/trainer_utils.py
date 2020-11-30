"""Utilities for trainer binary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

# Dependency imports
import time
import numpy as np
import six
from six.moves import zip

import ipdb

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import saver
from utils import parallel
from utils import datareader
from models import common_hparams
from models.transformer import Transformer

flags = tf.flags
FLAGS = flags.FLAGS
# TODO : I need to make this sync with the run hparam
flags.DEFINE_string("pretrain_output_dir", "../../../../data/external/LiSTra/train_models_enln/pretrain_model", "Base output directory for run.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("data_dir", "/tmp/data", "Directory with training data.")
flags.DEFINE_string("train_src_name", "2m.bpe.unk.zh", "src name of training data.")
flags.DEFINE_string("train_tgt_name", "2m.bpe.unk.en", "tgt name of training data.")
flags.DEFINE_string("vocab_src_name", "shared.vocab", "src name of vocab.")
flags.DEFINE_string("vocab_tgt_name", "shared.vocab", "tgt name of vocab.")
flags.DEFINE_integer("vocab_src_size", 30000, "source vocab size.")
flags.DEFINE_integer("vocab_tgt_size", 30000, "target vocab size.")

# Model
flags.DEFINE_string("model", "Transformer", "Which model to use.")
flags.DEFINE_string("hparams_set", "", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string("hparams", "", """A comma-separated list of `name=value` hyperparameter values.""")
flags.DEFINE_integer("train_steps", 250000, "The number of steps to run training for.")
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


def create_hparams():
    """Returns hyperparameters, including any flag value overrides.
    """
    if FLAGS.hparams_set == "transformer_params_base":
        hparams = common_hparams.transformer_params_base(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) ## !!
    elif FLAGS.hparams_set == "transformer_params_big":
        hparams = common_hparams.transformer_params_big(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name) ## !!
    elif FLAGS.hparams_set == "transformer_params_small":
        hparams = common_hparams.transformer_params_small(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name)  ## !!
#     elif FLAGS.hparams_set == "transformer_params_listra":
#         hparams = common_hparams.transformer_params_listra(FLAGS.data_dir, FLAGS.vocab_src_name, FLAGS.vocab_tgt_name)  ## !!
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
    train_input_fn = datareader.get_input_fn(
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
#         ipdb.set_trace()

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

    dev_input_fn = datareader.get_input_fn(
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

            tf.logging.info("Weight  %s\tshape    %s\tsize    %d",
                   v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
                   
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
    return parallel.Parallelism(
            datashard_devices,
            reuse=True,
            caching_devices=caching_devices,
            daisy_chain_variables=FLAGS.daisy_chain_variables)