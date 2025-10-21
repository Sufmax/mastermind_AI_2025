# train_batch.py (corrigé)
import tensorflow as tf
import numpy as np
import random
import os
import config
from policy import Policy
from transfer_utils import transfer_weights

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# ---------- helpers (inchangés / légèrement nettoyés) ----------
def batch_ints_to_digits_tensor(indices):
    indices = tf.cast(indices, tf.int32)
    # decode each integer into 4 base-6 digits
    def decode_one(x):
        return tf.stack([
            (x // (6**3)) % 6,
            (x // (6**2)) % 6,
            (x // (6**1)) % 6,
            (x // (6**0)) % 6
        ])
    return tf.map_fn(decode_one, indices, dtype=tf.int32)

def compute_feedback_batch(secret_digits, guess_digits):
    # secret_digits, guess_digits: (batch,4) ints in 0..5
    place = tf.reduce_sum(tf.cast(tf.equal(secret_digits, guess_digits), tf.int32), axis=1)  # (batch,)
    secret_oh = tf.one_hot(secret_digits, depth=6, dtype=tf.int32)  # (batch,4,6)
    guess_oh  = tf.one_hot(guess_digits,  depth=6, dtype=tf.int32)
    secret_counts = tf.reduce_sum(secret_oh, axis=1)  # (batch,6)
    guess_counts  = tf.reduce_sum(guess_oh, axis=1)   # (batch,6)
    common = tf.reduce_sum(tf.minimum(secret_counts, guess_counts), axis=1)  # (batch,)
    color = tf.cast(common, tf.int32)
    return color, place

def normalize_row_from_digits(guess_digits, color_norm, place_norm):
    guess_norm = tf.cast(guess_digits, tf.float32) / 5.0  # (batch,4)
    row = tf.concat([guess_norm, tf.expand_dims(color_norm,1), tf.expand_dims(place_norm,1)], axis=1)
    return row  # (batch,6)


# ---------- training function corrigée ----------
def train(num_steps=1000, batch_size=32, save_every=100, checkpoint_dir="checkpoints", policy=None):
    policy = policy or Policy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    ckpt = tf.train.Checkpoint(model=policy, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    max_len = config.max_episode_length
    eps = 1e-9

    for step in range(1, num_steps + 1):
        # sample batch secrets
        secrets = np.random.randint(0, config.max_guesses, size=(batch_size,), dtype=np.int32)
        secret_digits = batch_ints_to_digits_tensor(secrets)  # (batch,4)

        # initialize
        history = tf.zeros((batch_size, max_len, 6), dtype=tf.float32)
        lengths = tf.zeros((batch_size,), dtype=tf.int32)
        done = tf.zeros((batch_size,), dtype=tf.bool)

        logpi_ta = tf.TensorArray(tf.float32, size=max_len)
        reward_ta = tf.TensorArray(tf.float32, size=max_len)
        active_ta = tf.TensorArray(tf.float32, size=max_len)

        with tf.GradientTape() as tape:
            for t in range(max_len):
                mask = tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.int32)  # (batch, max_len)
                probs = policy(history, mask=mask, with_sigmoid=True)  # (batch,5,4)

                # sample bernoulli inside tape
                u = tf.random.uniform(shape=tf.shape(probs), minval=0.0, maxval=1.0)
                bern = tf.cast(u < probs, tf.float32)  # (batch,5,4)

                # log prob
                p_clip = tf.clip_by_value(probs, eps, 1.0 - eps)
                log_mat = bern * tf.math.log(p_clip) + (1 - bern) * tf.math.log(1 - p_clip)  # (batch,5,4)
                log_pi = tf.reduce_sum(log_mat, axis=[1,2])  # (batch,)

                # guess digits
                guess_digits = tf.cast(tf.reduce_sum(bern, axis=1), tf.int32)  # (batch,4)

                # feedback
                color_raw, place_raw = compute_feedback_batch(secret_digits, guess_digits)
                color_norm = tf.cast(color_raw, tf.float32) / 4.0
                place_norm = tf.cast(place_raw, tf.float32) / 4.0

                # reward: 1 on the step that finds the code, 0 else
                not_done = tf.logical_not(done)
                newly_done = tf.logical_and(not_done, tf.equal(place_raw, 4))
                reward = tf.cast(newly_done, tf.float32)  # 1 where finished on this step, else 0

                # update done mask
                done = tf.logical_or(done, newly_done)

                # update history rows only for episodes that were not done BEFORE this step
                row = normalize_row_from_digits(guess_digits, color_norm, place_norm)  # (batch,6)
                # update history[:, t, :] for sequences still active at this step
                not_done_float = tf.cast(not_done, tf.float32)  # (batch,)
                not_done_mask = tf.reshape(not_done_float, (batch_size, 1, 1))  # (batch,1,1)
                row_expanded = tf.expand_dims(row, axis=1)  # (batch,1,6)
                # write row_expanded into history[:, t:t+1, :] where not_done == 1
                before = history[:, :t, :]
                after = history[:, t+1:, :]
                to_write = tf.where(tf.cast(not_done_mask, tf.bool), row_expanded, tf.zeros_like(row_expanded))
                history = tf.concat([before, to_write, after], axis=1)

                # increment lengths for not_done episodes
                lengths += tf.cast(not_done, tf.int32)

                # write logs to TA
                logpi_ta = logpi_ta.write(t, log_pi)
                reward_ta = reward_ta.write(t, reward)
                active_ta = active_ta.write(t, tf.cast(not_done, tf.float32))

                if tf.reduce_all(done):
                    # fill remaining steps with zeros
                    for tt in range(t+1, max_len):
                        logpi_ta = logpi_ta.write(tt, tf.zeros((batch_size,), dtype=tf.float32))
                        reward_ta = reward_ta.write(tt, tf.zeros((batch_size,), dtype=tf.float32))
                        active_ta = active_ta.write(tt, tf.zeros((batch_size,), dtype=tf.float32))
                    break

            # stack
            logpi_stack = tf.transpose(logpi_ta.stack(), perm=[1,0])    # (batch, T)
            reward_stack = tf.transpose(reward_ta.stack(), perm=[1,0])  # (batch, T)
            active_stack = tf.transpose(active_ta.stack(), perm=[1,0])  # (batch, T)

            # compute returns G_t = sum_{k=t}^{T-1} reward_k (discount=1)
            # do cumsum from right
            rev_rewards = tf.reverse(reward_stack, axis=[1])
            rev_cumsum = tf.cumsum(rev_rewards, axis=1)
            returns = tf.reverse(rev_cumsum, axis=[1])  # (batch, T)

            # mask returns where inactive
            returns = returns * active_stack

            # normalize returns across all active entries (reduce variance)
            active_count = tf.reduce_sum(active_stack) + eps
            mean_returns = tf.reduce_sum(returns) / active_count
            # compute std across active entries
            sq = tf.square(returns - mean_returns) * active_stack
            std_returns = tf.sqrt(tf.reduce_sum(sq) / active_count + eps)
            returns_norm = (returns - mean_returns) / (std_returns + eps)  # (batch,T)

            # compute loss = - mean_{batch,t active} returns_norm * logpi
            loss_matrix = - returns_norm * logpi_stack * active_stack
            loss = tf.reduce_sum(loss_matrix) / active_count

        # grads & apply
        grads = tape.gradient(loss, policy.variables)
        optimizer.apply_gradients(zip(grads, policy.variables))

        # diagnostics: average length = mean(lengths) (float)
        avg_length = float(tf.reduce_mean(tf.cast(lengths, tf.float32)).numpy())

        if step % 10 == 0:
            print(f"Step {step}/{num_steps} loss={loss.numpy():.4f} avg_length={avg_length:.2f}")

        if step % save_every == 0 or step == num_steps:
            ckpt_manager.save()
            print(f"Checkpoint saved at step {step}")

    print("Training finished.")

if __name__ == "__main__":
    interactive_train()
