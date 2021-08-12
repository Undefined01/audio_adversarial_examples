# attack.py -- generate audio adversarial examples
##
# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from tensorflow.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"


class Attack:
    def __init__(
        self,
        max_audio_length,
        max_target_phrase_length,
        batch_size=1,
        l2penalty=float("inf"),
    ):
        # Basic information for building inference graph
        self.batch_size = batch_size
        self.max_target_phrase_length = max_target_phrase_length
        self.max_audio_length = max_audio_length

        # Variables
        with tfv1.variable_scope("adv", reuse=tfv1.AUTO_REUSE):
            self.delta = tf.Variable(
                tf.zeros(max_audio_length, dtype=tf.float32), name="delta"
            )
            self.rescale = tf.Variable(tf.ones((1,), dtype=tf.float32), name="rescale")
            self.lr = tf.Variable(1e2, shape=(), name="lr")
            self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Placeholder for inputs
        self.mask = tfv1.placeholder(
            shape=(batch_size, max_audio_length), dtype=tf.bool
        )
        self.audio = tfv1.placeholder(
            shape=(batch_size, max_audio_length), dtype=tf.float32
        )
        self.length = tfv1.placeholder(shape=(batch_size,), dtype=tf.int32)
        self.target_phrase = tfv1.placeholder(
            shape=(batch_size, max_target_phrase_length), dtype=tf.int32
        )
        self.target_phrase_length = tfv1.placeholder(
            shape=(batch_size,), dtype=tf.int32
        )

        # Prepare input audios
        offset = tf.random.uniform(
            (), minval=0, maxval=max_audio_length, dtype=tf.int32
        )
        apply_delta = tf.concat((self.delta[offset:], self.delta[:offset]), axis=0)
        apply_delta = tf.clip_by_value(apply_delta, -2000, 2000)
        apply_delta = apply_delta * self.rescale * tf.cast(self.mask, tf.float32)
        noise = tf.random.normal((batch_size, max_audio_length), stddev=2)
        self.noised_audio = tf.clip_by_value(
            self.audio + apply_delta + noise, -(2 ** 15), 2 ** 15 - 1
        )

        # Get inference result of DeepSpeech
        self.logits = get_logits(self.noised_audio, self.length)
        self.decoded, _ = tfv1.nn.ctc_beam_search_decoder(
            self.logits, self.length, merge_repeated=False, beam_width=100
        )

        # Calculate loss
        target = ctc_label_dense_to_sparse(
            self.target_phrase, self.target_phrase_length
        )
        self.ctc_loss = tfv1.nn.ctc_loss(
            labels=tf.cast(target, tf.int32),
            inputs=self.logits,
            sequence_length=self.length,
        )
        if not np.isinf(l2penalty):
            l2diff = tf.reduce_mean((self.noised_audio - self.audio) ** 2, axis=1)
            loss = l2diff + l2penalty * self.ctc_loss
        else:
            loss = self.ctc_loss
        self.loss = loss

        # Optimize step
        # self.lr = tfv1.train.exponential_decay(
        #     1e-6, global_step=global_step,
        #     decay_steps=10, decay_rate=2)
        self.optimizer = tfv1.train.AdamOptimizer(self.lr)
        self.train_pos_op = self.optimizer.minimize(
            self.loss, global_step=self.global_step, var_list=[self.delta]
        )
        self.train_neg_op = self.optimizer.minimize(
            -self.loss, global_step=self.global_step, var_list=[self.delta]
        )

        delta_loudness = tfv1.summary.scalar(
            "delta_loudness",
            20
            * tf.log(tf.math.reduce_max(tf.abs(self.delta * self.rescale)))
            / tf.log(10.0),
        )
        self.pos_summary = tfv1.summary.merge(
            [
                tfv1.summary.scalar("pos_loss", tf.math.reduce_mean(self.loss)),
                delta_loudness
            ]
        )
        self.neg_summary = tfv1.summary.merge(
            [
                tfv1.summary.scalar("neg_loss", tf.math.reduce_mean(self.loss)),
                delta_loudness
            ]
        )

    def init_sess(self, sess, restore_path=None):
        saver = tfv1.train.Saver(
            set(tfv1.global_variables()) - set(tfv1.global_variables("adv"))
        )
        saver.restore(sess, restore_path)

        sess.run(
            tfv1.variables_initializer(
                self.optimizer.variables() + tf.global_variables("adv")
            )
        )
        sess.run(self.global_step.assign(0))

    def build_feed_dict(self, audios, lengths, target=None):
        assert audios.shape[0] == self.batch_size
        assert audios.shape[1] == self.max_audio_length
        assert lengths.shape[0] == self.batch_size

        masks = np.zeros((lengths.shape[0], self.max_audio_length), dtype=np.bool)
        for i, l in enumerate(lengths):
            masks[i, :l] = True

        feed_dict = {
            self.audio: audios,
            self.length: (lengths - 1) // 320,
            self.mask: masks,
        }

        if target is not None:
            padded_target = [
                [toks.index(x) for x in phrase.lower()]
                + [toks.index("-")] * (self.max_target_phrase_length - len(phrase))
                for phrase in target
            ]
            feed_dict = {
                **feed_dict,
                self.target_phrase: padded_target,
                self.target_phrase_length: [len(x) for x in target],
            }

        return feed_dict

    def inference(self, sess, feed_dict):
        out, logits = sess.run((self.decoded, self.logits), feed_dict=feed_dict)

        res = np.zeros(out[0].dense_shape) + len(toks) - 1
        for ii in range(len(out[0].values)):
            x, y = out[0].indices[ii]
            res[x, y] = out[0].values[ii]

        # the strings that are recognized.
        res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]

        # the argmax of the alignment.
        res2 = [
            "".join(toks[int(x)] for x in y[:l])
            for y, l in zip(np.argmax(logits, axis=2).T, feed_dict[self.length])
        ]
        return res, res2

    def train_step(self, sess, feed_dict, minimize_loss=True, summary_writer=None):
        train_op = self.train_pos_op if minimize_loss else self.train_neg_op
        summary_op = self.pos_summary if minimize_loss else self.neg_summary
        res = sess.run(
            (self.global_step, summary_op, self.loss, train_op), feed_dict=feed_dict
        )
        global_step, summary, loss, _ = res
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step)
        return loss

    def get_delta(self, sess):
        return sess.run(tf.clip_by_value(self.delta, -2000, 2000) * self.rescale)
