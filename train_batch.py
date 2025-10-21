# train_batch.py
import tensorflow as tf
import numpy as np
import random
import os
import argparse
import config
from policy import Policy

# Helpers (secret encoding/decoding)
def int_to_digits(index):
    # returns list length 4 of ints 0..5
    digits = []
    for i in range(4):
        power = 6 ** (3 - i)
        d = (index // power) % 6
        digits.append(d)
    return digits

def batch_ints_to_digits_tensor(indices):
    # indices: (batch,) int numpy or tf
    indices = tf.cast(indices, tf.int32)
    def decode_one(x):
        # returns (4,)
        res = []
        v = x
        for i in range(4):
            power = 6 ** (3 - i)
            res.append((v // power) % 6)
            v = tf.math.floordiv(v, power)
        return tf.stack(res)
    return tf.map_fn(lambda x: tf.stack([ (x // (6**3)) % 6,
                                          (x // (6**2)) % 6,
                                          (x // (6**1)) % 6,
                                          (x // (6**0)) % 6 ]), indices, dtype=tf.int32)

def binary_matrix_to_guess_digits_tf(binary):
    # binary: (batch,5,4) float32 zeros/ones -> sum rows -> (batch,4) ints
    counts = tf.reduce_sum(binary, axis=1)   # (batch,4)
    # round to nearest int just in case (should be integer already)
    return tf.cast(tf.round(counts), tf.int32)

def compute_feedback_batch(secret_digits, guess_digits):
    # secret_digits, guess_digits: (batch,4) ints in 0..5
    # place: sum over positions equal
    place = tf.reduce_sum(tf.cast(tf.equal(secret_digits, guess_digits), tf.int32), axis=1)  # (batch,)
    # color: multiset intersection
    # one-hot along last dim 6 -> sum positions -> counts (batch,6)
    secret_onehot = tf.reduce_sum(tf.one_hot(secret_digits, depth=6, dtype=tf.int32), axis=1)  # WRONG: shape adjust below
    # Above line is wrong due to shapes; use below robust build:
    secret_oh = tf.one_hot(secret_digits, depth=6, dtype=tf.int32)  # (batch,4,6)
    guess_oh  = tf.one_hot(guess_digits,  depth=6, dtype=tf.int32)
    secret_counts = tf.reduce_sum(secret_oh, axis=1)  # (batch,6)
    guess_counts  = tf.reduce_sum(guess_oh,  axis=1)  # (batch,6)
    common = tf.reduce_sum(tf.minimum(secret_counts, guess_counts), axis=1)  # (batch,)
    color = tf.cast(common, tf.int32)
    return color, place

def normalize_row_from_digits(guess_digits, color_norm, place_norm):
    # guess_digits: (batch,4) ints -> normalize /5.0
    guess_norm = tf.cast(guess_digits, tf.float32) / 5.0  # (batch,4)
    # concat to shape (batch,6)
    row = tf.concat([guess_norm, tf.expand_dims(color_norm,1), tf.expand_dims(place_norm,1)], axis=1)
    return row  # (batch,6)

def train(num_steps=10000, batch_size=32, save_every=100, checkpoint_dir="checkpoints"):
    policy = Policy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.reinforce_alpha)
    ckpt = tf.train.Checkpoint(model=policy, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    max_len = config.max_episode_length
    eps = 1e-9

    # training loop (each iteration produces one batch of episodes)
    for step in range(1, num_steps + 1):
        # sample batch of secrets (ints 0..6^4-1)
        secrets = np.random.randint(0, config.max_guesses, size=(batch_size,), dtype=np.int32)
        secret_digits = batch_ints_to_digits_tensor(secrets)  # (batch,4)

        # initialize history and masks
        history = tf.zeros((batch_size, max_len, 6), dtype=tf.float32)
        lengths = tf.zeros((batch_size,), dtype=tf.int32)
        done = tf.zeros((batch_size,), dtype=tf.bool)

        # we'll store log_pi per time per batch
        logpi_ta = tf.TensorArray(tf.float32, size=max_len, dynamic_size=False, infer_shape=True)
        active_ta = tf.TensorArray(tf.float32, size=max_len, dynamic_size=False, infer_shape=True)

        with tf.GradientTape() as tape:
            for t in range(max_len):
                # mask for policy: ones for positions < current length
                mask = tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.int32)  # (batch, max_len)
                # policy expects mask as numeric boolean/int - fine
                probs = policy(history, mask=mask, with_sigmoid=True)  # (batch,5,4)

                # sample bernoulli inside tape (so gradients flow to probs via log_pi)
                u = tf.random.uniform(shape=tf.shape(probs), minval=0.0, maxval=1.0)
                bern = tf.cast(u < probs, tf.float32)  # (batch,5,4)

                # compute log prob
                p_clip = tf.clip_by_value(probs, eps, 1.0 - eps)
                log_mat = bern * tf.math.log(p_clip) + (1 - bern) * tf.math.log(1 - p_clip)  # (batch,5,4)
                log_pi = tf.reduce_sum(log_mat, axis=[1,2])  # (batch,)

                # compute guess digits from bern (counts per column)
                guess_digits = tf.cast(tf.round(tf.reduce_sum(bern, axis=1)), tf.int32)  # (batch,4)

                # compute feedback (color, place)
                color_raw, place_raw = compute_feedback_batch(secret_digits, guess_digits)
                color_norm = tf.cast(color_raw, tf.float32) / 4.0
                place_norm = tf.cast(place_raw, tf.float32) / 4.0

                # reward: -1 for active episodes that are not yet done; 0 for done episodes
                not_done = tf.logical_not(done)
                reward = tf.where(not_done, -1.0, 0.0)  # (batch,)

                # mark newly finished episodes where place_raw == 4
                newly_done = tf.logical_and(not_done, tf.equal(place_raw, 4))
                done = tf.logical_or(done, newly_done)

                # update history and lengths only for episodes that were not done (i.e., those that actually played this turn)
                # we create row for all batches, but only write for those that were not done
                row = normalize_row_from_digits(guess_digits, color_norm, place_norm)  # (batch,6)
                # update history at time t: history[:, t, :] = row for those not done
                # need to convert to tensor updates
                update_mask = tf.expand_dims(tf.cast(not_done, tf.float32), axis=1)  # (batch,1)
                row_expanded = tf.expand_dims(row, axis=1)  # (batch,1,6)
                history = tf.concat([
                    history[:, :t, :],
                    tf.where(tf.cast(update_mask[:,:,None], tf.bool), row_expanded, tf.zeros_like(row_expanded)),
                    history[:, t+1:, :]
                ], axis=1)

                # increment lengths for episodes that were not done
                lengths += tf.cast(not_done, tf.int32)

                # store log_pi and active mask (1 if not_done else 0) for this timestep
                logpi_ta = logpi_ta.write(t, log_pi)
                active_ta = active_ta.write(t, tf.cast(not_done, tf.float32))

                # stop early if all done
                if tf.reduce_all(done):
                    # fill remaining time steps with zeros in TA
                    for tt in range(t+1, max_len):
                        logpi_ta = logpi_ta.write(tt, tf.zeros((batch_size,), dtype=tf.float32))
                        active_ta = active_ta.write(tt, tf.zeros((batch_size,), dtype=tf.float32))
                    break

            # stack TAs into (T, batch) -> transpose to (batch, T)
            logpi_stack = tf.transpose(logpi_ta.stack(), perm=[1,0])   # (batch, T)
            active_stack = tf.transpose(active_ta.stack(), perm=[1,0]) # (batch, T)

            # lengths: actual steps per episode (vector batch)
            L = tf.cast(lengths, tf.float32)  # (batch,)
            T_actual = tf.shape(logpi_stack)[1]

            # build returns per batch per time: G_t = -(L - t) for t in [0..T_actual-1], but only where active==1
            # create times vector t = [0,1,2,...,T_actual-1]
            times = tf.cast(tf.range(T_actual), tf.float32)  # (T,)
            # Expand: L[:,None] - times[None,:] -> (batch, T)
            returns = -( (tf.expand_dims(L,1) - tf.reshape(times, (1,-1))) )  # (batch, T)
            # For times >= L, should be irrelevant; mask them by active_stack
            returns = returns * active_stack  # zeroed where inactive

            # compute loss: - sum_b sum_t G_bt * logpi_bt
            loss_matrix = - (returns * logpi_stack) * active_stack
            loss = tf.reduce_sum(loss_matrix) / tf.cast(batch_size, tf.float32)

        # gradients
        grads = tape.gradient(loss, policy.variables)
        optimizer.apply_gradients(zip(grads, policy.variables))

        if step % 10 == 0:
            print(f"Step {step}/{num_steps} loss={loss.numpy():.4f} avg_length={tf.reduce_mean(L).numpy():.2f}")

        if step % save_every == 0 or step == num_steps:
            ckpt_manager.save()
            print(f"Checkpoint saved at step {step}")

    print("Training finished.")
