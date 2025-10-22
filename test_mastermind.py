import tensorflow as tf
import numpy as np
import os
import config
from policy import Policy
from episode import Episode

def load_policy_from_checkpoint(ckpt_dir, ckpt_num):
    # Compose les chemins des fichiers
    base = os.path.join(ckpt_dir, f"ckpt-{ckpt_num}")
    data_file = base + ".data-00000-of-00001"
    index_file = base + ".index"
    if not (os.path.exists(data_file) and os.path.exists(index_file)):
        print(f"❌ Fichiers de checkpoint non trouvés :\n  {data_file}\n  {index_file}")
        return None

    # Crée une instance de Policy et charge les poids
    policy = Policy()
    # Appel fictif pour construire les poids
    policy(tf.zeros((1, config.max_episode_length, 6)))
    ckpt = tf.train.Checkpoint(model=policy)
    status = ckpt.restore(base)
    status.expect_partial()
    print(f"✅ Checkpoint chargé depuis {base}")
    return policy

def print_policy_info(policy):
    print("=== Infos sur le modèle chargé ===")
    print(f"LSTM hidden size : {policy.lstm.units}")
    print(f"Paramètres totaux : {np.sum([np.prod(v.shape) for v in policy.variables])}")
    print("==================================")

def ask_secret():
    while True:
        s = input("➡️  Entrez la combinaison secrète (4 chiffres entre 0 et 5, ex: 1234) : ").strip()
        if len(s) == 4 and all(c in "012345" for c in s):
            return [int(c) for c in s]
        print("❌ Entrée invalide. Essayez encore.")

def print_step(step, guess, feedback_color, feedback_place):
    guess_str = "".join(str(d) for d in guess)
    print(f"Tour {step+1:2d} | Proposition IA : {guess_str} | Couleurs bien placées : {int(feedback_place*4)}/4 | Bonnes couleurs (total) : {int(feedback_color*4)}/4")

def main():
    print("=== 🧪 Test interactif de l'IA Mastermind ===")
    ckpt_dir = input("➡️  Dossier de sauvegarde (défaut=checkpoints): ").strip() or "checkpoints"
    ckpt_num = input("➡️  Numéro de backup à charger (ex: 100) : ").strip()
    if not ckpt_num.isdigit():
        print("❌ Numéro de backup invalide.")
        return
    policy = load_policy_from_checkpoint(ckpt_dir, ckpt_num)
    if policy is None:
        return
    print_policy_info(policy)

    while True:
        secret = ask_secret()
        episode = Episode(policy, secret)
        steps = episode.generate()
        print("\n--- Déroulement de la partie ---")
        for i, step in enumerate(steps):
            print_step(i, step['guess_digits'], step['feedback_color'], step['feedback_place'])
            if step['feedback_place'] == 1.0:
                print(f"🎉 L'IA a trouvé la combinaison en {i+1} tours !\n")
                break
        else:
            print("❌ L'IA n'a pas trouvé la combinaison en 30 tours.\n")

        again = input("Voulez-vous tester une autre combinaison ? (o/n) : ").strip().lower()
        if again != "o":
            print("Fin du test.")
            break

if __name__ == "__main__":
    main()
