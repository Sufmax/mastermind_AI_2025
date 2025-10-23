import tensorflow as tf
import numpy as np
import os
import config
from policy import Policy
from transfer_utils import transfer_weights

# --- Fonctions Helper (inchangées) ---
def batch_ints_to_digits_tensor(indices):
    indices = tf.cast(indices, tf.int32)
    powers = 6 ** tf.constant([3, 2, 1, 0], dtype=tf.int32)
    digits = tf.expand_dims(indices, axis=-1) // powers
    return digits % 6
def binary_matrix_to_guess_digits_tf(binary):
    counts = tf.reduce_sum(binary, axis=1)
    return tf.cast(tf.round(counts), tf.int32)
def compute_feedback_batch(secret_digits, guess_digits):
    place = tf.reduce_sum(tf.cast(tf.equal(secret_digits, guess_digits), tf.int32), axis=1)
    secret_oh = tf.one_hot(secret_digits, depth=6, dtype=tf.int32)
    guess_oh = tf.one_hot(guess_digits, depth=6, dtype=tf.int32)
    secret_counts = tf.reduce_sum(secret_oh, axis=1)
    guess_counts = tf.reduce_sum(guess_oh, axis=1)
    common = tf.reduce_sum(tf.minimum(secret_counts, guess_counts), axis=1)
    return tf.cast(common, tf.int32), place
def normalize_row_from_digits(guess_digits, color_norm, place_norm):
    guess_norm = tf.cast(guess_digits, tf.float32) / 5.0
    return tf.concat([guess_norm, tf.expand_dims(color_norm, 1), tf.expand_dims(place_norm, 1)], axis=1)

# --- Interface Utilisateur (inchangée, mais le code qui suit doit marcher) ---
def interactive_train():
    print("=== 🎯 Entraînement IA Mastermind ===")
    ckpt_path = input("➡️  Chemin du checkpoint (laisser vide pour un nouveau modèle): ").strip()
    old_policy, lstm_size_loaded = None, None
    if ckpt_path and os.path.exists(ckpt_path + ".index"):
        print("Chargement du modèle existant...")
        try:
            # L'ancien modèle est aussi un keras.Model maintenant
            old_policy = Policy() 
            # Pour charger les poids, le modèle doit être construit.
            # On le restaure vide, puis on le charge.
            ckpt = tf.train.Checkpoint(model=old_policy)
            ckpt.restore(ckpt_path).expect_partial()
            # La taille peut être déduite après un appel
            dummy_input = (tf.zeros((1, 1, 6)), tf.zeros((1, 1), dtype=tf.bool))
            _ = old_policy(dummy_input)
            lstm_size_loaded = old_policy.lstm.units
            print(f"✅ Modèle chargé avec LSTM de taille {lstm_size_loaded}")
        except Exception as e:
            print(f"⚠️ Impossible de charger le checkpoint : {e}")
    else:
        print("Aucun checkpoint trouvé, un nouveau modèle sera créé.")

    new_size_str = input(f"➡️  Nouvelle taille LSTM (actuelle {lstm_size_loaded or config.lstm_hidden_size}): ").strip()
    new_size = int(new_size_str) if new_size_str else (lstm_size_loaded or config.lstm_hidden_size)
    new_policy = Policy(lstm_hidden_size=new_size)
    
    # Construire le modèle pour pouvoir transférer les poids
    dummy_input = (tf.zeros((1, 1, 6)), tf.zeros((1, 1), dtype=tf.bool))
    _ = new_policy(dummy_input)

    if old_policy and new_size != lstm_size_loaded:
        print("Transfert des poids vers le nouveau modèle...")
        transfer_weights(old_policy, new_policy)
    elif old_policy:
        print("Taille identique, modèle restauré directement.")
        new_policy.set_weights(old_policy.get_weights())
    else:
        print("Nouveau modèle initialisé aléatoirement.")

    def ask(prompt, default, typ):
        s = input(f"➡️  {prompt} (défaut={default}) : ").strip()
        return typ(s) if s != "" else default
    num_steps = ask("Nombre d'étapes", 10000, int)
    batch_size = ask("Taille du batch", 32, int)
    save_every = ask("Sauvegarde toutes les X étapes", 100, int)
    checkpoint_dir = input("➡️  Dossier de sauvegarde (défaut=checkpoints): ").strip() or "checkpoints"
    learning_rate = ask("Learning rate", config.reinforce_alpha, float)
    log_every = ask("Afficher les logs toutes les X étapes", 10, int)
    print(f"\n🚀 Démarrage de l'entraînement ({num_steps} étapes, batch={batch_size})\n")
    train(num_steps=num_steps, batch_size=batch_size, save_every=save_every, log_every=log_every, checkpoint_dir=checkpoint_dir, policy=new_policy, learning_rate=learning_rate)


# --- Logique d'Entraînement ---
# @tf.function # Garder commenté pour le débogage initial
def train_step(policy, optimizer, secrets, batch_size):
    max_len, eps = config.max_episode_length, 1e-9
    secret_digits = batch_ints_to_digits_tensor(secrets)
    history = tf.zeros((batch_size, max_len, 6), dtype=tf.float32)
    lengths = tf.zeros((batch_size,), dtype=tf.int32)
    done = tf.zeros((batch_size,), dtype=tf.bool)
    prev_place_norm = tf.zeros((batch_size,), dtype=tf.float32)
    prev_color_norm = tf.zeros((batch_size,), dtype=tf.float32)
    logpi_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    rewards_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    with tf.GradientTape() as tape:
        for t in tf.range(max_len):
            if tf.reduce_all(done): break

            mask = tf.sequence_mask(lengths, maxlen=max_len, dtype=tf.int32)
            
            # === MODIFICATION DE L'APPEL AU MODÈLE ===
            probs = policy((history, mask), with_sigmoid=True)
            
            p_clip = tf.clip_by_value(probs, eps, 1.0 - eps)
            bern = tf.cast(tf.random.uniform(tf.shape(p_clip)) < p_clip, tf.float32)
            # L'astuce est d'utiliser stop_gradient sur la partie de bern qui multiplie
            # pour que le gradient ne passe que par log(p) et pas par l'échantillon.
            # grad(bern * log(p)) -> grad(log(p))
            log_mat = bern * tf.math.log(p_clip) + (1.0 - bern) * tf.math.log(1.0 - p_clip)
            log_pi = tf.reduce_sum(log_mat, axis=[1, 2])

            guess_digits = binary_matrix_to_guess_digits_tf(tf.stop_gradient(bern)) # On ne dérive pas à travers les digits
            color_raw, place_raw = compute_feedback_batch(secret_digits, guess_digits)
            color_norm = tf.cast(color_raw, tf.float32) / 4.0
            place_norm = tf.cast(place_raw, tf.float32) / 4.0
            not_done = tf.logical_not(done)
            
            reward = -1.0
            delta_place = place_norm - prev_place_norm
            reward += delta_place * 2.0
            delta_color = color_norm - prev_color_norm
            reward += delta_color * 0.5
            final_reward = tf.where(not_done, reward, 0.0)
            rewards_ta = rewards_ta.write(t, final_reward)
            
            prev_place_norm = tf.where(not_done, place_norm, prev_place_norm)
            prev_color_norm = tf.where(not_done, color_norm, prev_color_norm)

            newly_done = tf.logical_and(not_done, tf.equal(place_raw, 4))
            done = tf.logical_or(done, newly_done)
            row = normalize_row_from_digits(guess_digits, color_norm, place_norm)
            indices_to_update = tf.where(not_done)
            history = tf.tensor_scatter_nd_update(history, tf.stack([indices_to_update[:, 0], tf.cast(tf.fill(tf.shape(indices_to_update)[0], t), dtype=tf.int64)], axis=1), tf.gather_nd(row, indices_to_update))
            lengths += tf.cast(not_done, tf.int32)
            logpi_ta = logpi_ta.write(t, log_pi)

        logpi_stack = tf.transpose(logpi_ta.stack(), perm=[1, 0])
        rewards_stack = tf.transpose(rewards_ta.stack(), perm=[1, 0])
        active_mask = tf.sequence_mask(lengths, maxlen=tf.shape(rewards_stack)[1], dtype=tf.float32)
        returns = tf.reverse(tf.math.cumsum(tf.reverse(rewards_stack, axis=[1]), axis=1), axis=[1])
        
        # L'objectif de REINFORCE est de maximiser E[G_t * log(pi)].
        # Pour un minimiseur, on minimise -E[G_t * log(pi)].
        objective = returns * logpi_stack
        loss = -tf.reduce_sum(objective * active_mask) / tf.cast(batch_size, tf.float32)

    grads = tape.gradient(loss, policy.trainable_variables)
    
    # Vérification pour le débogage
    if grads is None or all(g is None for g in grads):
         print("ERREUR: Aucun gradient calculé. Le graphe est probablement déconnecté.")
         return None, None, None

    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss, lengths, tf.reduce_mean(tf.reduce_sum(rewards_stack, axis=1))


def train(num_steps=10000, batch_size=32, save_every=100, log_every=10, checkpoint_dir="checkpoints", policy=None, learning_rate=config.reinforce_alpha):
    policy = policy or Policy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ckpt = tf.train.Checkpoint(model=policy, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    for step in range(1, num_steps + 1):
        secrets = np.random.randint(0, config.max_guesses, size=(batch_size,), dtype=np.int32)
        loss, lengths, avg_total_reward = train_step(policy, optimizer, secrets, batch_size)
        if loss is None: break
        if step % log_every == 0:
            avg_len = tf.reduce_mean(tf.cast(lengths, tf.float32))
            print(f"Step {step}/{num_steps} loss={-loss.numpy():.4f} avg_length={avg_len.numpy():.2f} avg_ep_reward={avg_total_reward.numpy():.2f}")
        if step % save_every == 0 or step == num_steps:
            save_path = ckpt_manager.save()
            print(f"Checkpoint sauvegardé à l'étape {step} : {save_path}")
    print("✅ Entraînement terminé.")


if __name__ == "__main__":
    interactive_train()
