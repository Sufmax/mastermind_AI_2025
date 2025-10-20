import tensorflow as tf
import random
import os
import sys
from policy import Policy
from episode import Episode
import config

def train(num_episodes=1000, save_every=100, checkpoint_dir="checkpoints"):
    pol = Policy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.reinforce_alpha)

    # checkpoint manager
    ckpt = tf.train.Checkpoint(model=pol, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    for j in range(1, num_episodes + 1):
        random_secret = random.randint(0, config.max_guesses - 1)
        e = Episode(pol, random_secret)
        history = e.generate()

        print("Episode length:", len(history))

        G = -1  # return

        for i in reversed(range(1, len(history))):
            history_so_far = history[:i]
            next_action, _ = history[i]

            with tf.GradientTape() as tape:
                logits = pol(history_so_far, with_softmax=False)
                labels = tf.one_hot([next_action], config.max_guesses)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

            grads = tape.gradient(loss, pol.variables)
            optimizer.apply_gradients(zip(grads, pol.variables))
            G -= 1

            sys.stdout.write("{}/{}\r".format(len(history)-i, len(history)))
            sys.stdout.flush()

        # save checkpoint
        if j % save_every == 0 or j == num_episodes:
            ckpt_manager.save()
            print(f"\nCheckpoint saved at episode {j}")

if __name__ == "__main__":
    train()
