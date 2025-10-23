import tensorflow as tf
import numpy as np
import random
import os
import config
from policy import Policy
from transfer_utils import transfer_weights

def int_to_digits(index):
    digits = []
    for i in range(4):
        power = 6 ** (3 - i)
        d = (index // power) % 6
        digits.append(d)
    return digits

def batch_ints_to_digits_tensor(indices):
    indices = np.array(indices).astype(np.int32)
    digits = np.zeros((len(indices), 4), dtype=np.int32)
    for i, idx in enumerate(indices):
        for j in range(4):
            power = 6 ** (3 - j)
            digits[i, j] = (idx // power) % 6
    return tf.convert_to_tensor(digits, dtype=tf.int32)

def binary_matrix_to_guess_digits_tf(binary):
    counts = tf.reduce_sum(binary, axis=1)
    return tf.cast(tf.round(counts), tf.int32)

def compute_feedback_batch(secret_digits, guess_digits):
    place = tf.reduce_sum(tf.cast(tf.equal(secret_digits, guess_digits), tf.int32), axis=1)
    secret_oh = tf.one_hot(secret_digits, depth=6, dtype=tf.int32)
    guess_oh  = tf.one_hot(guess_digits,  depth=6, dtype=tf.int32)
    secret_counts = tf.reduce_sum(secret_oh, axis=1)
    guess_counts  = tf.reduce_sum(guess_oh,  axis=1)
    common = tf.reduce_sum(tf.minimum(secret_counts, guess_counts), axis=1)
    color = tf.cast(common, tf.int32)
    return color, place

def normalize_row_from_digits(guess_digits, color_norm, place_norm):
    guess_norm = tf.cast(guess_digits, tf.float32) / 5.0
    row = tf.concat([guess_norm, tf.expand_dims(color_norm,1), tf.expand_dims(place_norm,1)], axis=1)
    return row

def interactive_train():
    print("=== 🎯 Entraînement IA Mastermind ===")

    ckpt_path = input("➡️  Entrez le chemin du checkpoint à charger (ou laissez vide pour créer un nouveau): ").strip()
    old_policy = None
    lstm_size_loaded = None

    if ckpt_path and os.path.exists(ckpt_path):
        print("Chargement du modèle existant...")
        try:
            old_policy = Policy()
            ckpt = tf.train.Checkpoint(model=old_policy)
            ckpt.restore(ckpt_path).expect_partial()
            lstm_size_loaded = old_policy.lstm.units
            print(f"✅ Modèle chargé avec LSTM de taille {lstm_size_loaded}")
        except Exception as e:
            print("⚠️ Impossible de charger le checkpoint :", e)
    else:
        print("Aucun checkpoint trouvé, un nouveau modèle sera créé.")

    new_size_str = input(f"➡️  Nouvelle taille LSTM (actuelle {lstm_size_loaded or config.lstm_hidden_size}): ").strip()
    if new_size_str == "":
        new_size = lstm_size_loaded or config.lstm_hidden_size
    else:
        new_size = int(new_size_str)

    new_policy = Policy(lstm_hidden_size=new_size)

    if old_policy and new_size != lstm_size_loaded:
        # Après avoir créé old_policy et new_policy
        dummy_input = tf.zeros((1, config.max_episode_length, 6), dtype=tf.float32)
        old_policy(dummy_input)
        new_policy(dummy_input)
        print("Transfert des poids vers le nouveau modèle...")
        transfer_weights(old_policy, new_policy)
    elif old_policy:
        print("Taille identique, modèle restauré directement.")
        new_policy = old_policy
    else:
        print("Nouveau modèle initialisé aléatoirement.")

    def ask(prompt, default, typ):
        s = input(f"➡️  {prompt} (défaut={default}) : ").strip()
        return typ(s) if s != "" else default

    num_steps = ask("Nombre d'étapes", 10000, int)
    batch_size = ask("Taille du batch", 32, int)
    save_every = ask("Sauvegarde toutes les X étapes", 100, int)
    checkpoint_dir = input("➡️  Dossier de sauvegarde (défaut=checkpoints): ").strip() or "checkpoints"
    reinforce_alpha= ask("Le learning rate (plus c'est bas, plus c'est précis)", config.reinforce_alpha, float)

    print(f"\n🚀 Démarrage de l'entraînement ({num_steps} étapes, batch={batch_size})\n")
    train(num_steps=num_steps, batch_size=batch_size, save_every=save_every, checkpoint_dir=checkpoint_dir, policy=new_policy, reinforce_alpha=reinforce_alpha)

def train(num_steps=10000, batch_size=32, save_every=100, checkpoint_dir="checkpoints", policy=None, reinforce_alpha=config.reinforce_alpha):
    policy = policy or Policy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=reinforce_alpha) #AVANT: optimizer = tf.keras.optimizers.SGD(learning_rate=reinforce_alpha)
    ckpt = tf.train.Checkpoint(model=policy, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    max_len = config.max_episode_length
    eps = 1e-9

    for step in range(1, num_steps + 1):
        secrets = np.random.randint(0, config.max_guesses, size=(batch_size,), dtype=np.int32)
        secret_digits = batch_ints_to_digits_tensor(secrets)

        history = tf.zeros((batch_size, max_len, 6), dtype=tf.float32)
        lengths = tf.zeros((batch_size,), dtype=tf.int32)
        done = tf.zeros((batch_size,), dtype=tf.bool)

        logpi_ta = tf.TensorArray(tf.float32, size=max_len)
        active_ta = tf.TensorArray(tf.float32, size=max_len)

        with tf.GradientTape() as tape:
            prev_place_norm = tf.zeros((batch_size,), dtype=tf.float32)
            prev_color_norm = tf.zeros((batch_size,), dtype=tf.float32)
            rewards_ta = tf.TensorArray(tf.float32, size=max_len)
            for t in range(max_len):
                mask = tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.int32)
                probs = policy(history, mask=mask, with_sigmoid=True)


                p_clip = tf.clip_by_value(probs, eps, 1.0 - eps)
                u = tf.random.uniform(tf.shape(probs))
                bern = tf.cast(u < p_clip, tf.float32)
                log_mat = bern * tf.math.log(p_clip) + (1 - bern) * tf.math.log(1 - p_clip)
                
                log_pi = tf.reduce_sum(log_mat, axis=[1,2])

                guess_digits = tf.cast(tf.round(tf.reduce_sum(bern, axis=1)), tf.int32)
                color_raw, place_raw = compute_feedback_batch(secret_digits, guess_digits)
                color_norm = tf.cast(color_raw, tf.float32) / 4.0
                place_norm = tf.cast(place_raw, tf.float32) / 4.0

                not_done = tf.logical_not(done)
                newly_done = tf.logical_and(not_done, tf.equal(place_raw, 4))
                done = tf.logical_or(done, newly_done)
                
                # --- Reward shaping ---
                # delta_place = place_norm - prev_place_norm
                # delta_color = color_norm - prev_color_norm
                delta_place = place_norm - prev_place_norm
                delta_color = color_norm - prev_color_norm
                # On ne donne le bonus qu'aux batchs actifs
                reward = -1.0 + 0.5 * delta_place + 0.2 * delta_color
                reward = tf.clip_by_value(reward, -1.0, 1.0)
                reward = tf.where(not_done, reward, 0.0)
                rewards_ta = rewards_ta.write(t, reward)
                
                # Met à jour prev_* pour le prochain tour
                prev_place_norm = tf.where(not_done, place_norm, prev_place_norm)
                prev_color_norm = tf.where(not_done, color_norm, prev_color_norm)
                
                row = normalize_row_from_digits(guess_digits, color_norm, place_norm)
                indices_to_update = tf.where(tf.cast(not_done, tf.bool))[:, 0]
                history = tf.tensor_scatter_nd_update(
                    history, 
                    tf.stack([indices_to_update, tf.cast(tf.fill(tf.shape(indices_to_update), t), dtype=tf.int64)], axis=1), 
                    tf.gather(row, indices_to_update)
                )
                
                lengths += tf.cast(not_done, tf.int32)

                logpi_ta = logpi_ta.write(t, log_pi)
                active_ta = active_ta.write(t, tf.cast(not_done, tf.float32))

                if tf.reduce_all(done):
                    break

            logpi_stack = tf.transpose(logpi_ta.stack(), perm=[1,0])
            active_stack = tf.transpose(active_ta.stack(), perm=[1,0])

            L = tf.cast(lengths, tf.float32)
            T_actual = tf.shape(logpi_stack)[1]
            times = tf.cast(tf.range(T_actual), tf.float32)
            # returns = -(L - t - 1)
            ##returns = -((tf.expand_dims(L,1) - tf.reshape(times, (1,-1)) - 1))
            ##returns *= active_stack

            # rewards_stack: (batch, max_len)
            rewards_stack = tf.transpose(rewards_ta.stack(), perm=[1,0])  # (batch, max_len)
            
            returns = tf.reverse(tf.math.cumsum(tf.reverse(rewards_stack, axis=[1]), axis=1), axis=[1])
            returns *= active_stack

            # LOSS POSITIVE À MINIMISER
            loss = -tf.reduce_sum((returns * logpi_stack) * active_stack) / tf.cast(batch_size, tf.float32)

            if tf.math.reduce_any(tf.math.is_nan(loss)):
                print("\nNaN detected in loss! Stopping training.\n")
                return

        grads = tape.gradient(loss, policy.variables)
        optimizer.apply_gradients(zip(grads, policy.variables))

        if step % 10 == 0:
            print(f"Step {step}/{num_steps} loss={-loss.numpy():.4f} avg_length={tf.reduce_mean(L).numpy():.2f}")

        if step % save_every == 0 or step == num_steps:
            ckpt_manager.save()
            print(f"Checkpoint sauvegardé à l'étape {step}")

    print("✅ Entraînement terminé.")

if __name__ == "__main__":
    interactive_train()
