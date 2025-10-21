# transfer_utils.py
import tensorflow as tf
import numpy as np
from policy import Policy

def transfer_weights(old_policy, new_policy):
    """
    Copie les poids de old_policy vers new_policy, même si lstm_hidden_size est différent.
    Les nouvelles dimensions sont initialisées aléatoirement, mais le bloc commun est copié.
    """

    # LSTM
    old_w = old_policy.lstm.get_weights()
    new_w = new_policy.lstm.get_weights()

    old_kernel, old_recurrent, old_bias = old_w
    new_kernel, new_recurrent, new_bias = new_w

    min_u = min(old_recurrent.shape[0], new_recurrent.shape[0])

    # Copier uniquement la partie commune
    new_kernel[:, :4*min_u] = old_kernel[:, :4*min_u]
    new_recurrent[:min_u, :4*min_u] = old_recurrent[:min_u, :4*min_u]
    new_bias[:4*min_u] = old_bias[:4*min_u]

    new_policy.lstm.set_weights([new_kernel, new_recurrent, new_bias])

    # Dense
    old_k, old_b = old_policy.out.get_weights()
    new_k, new_b = new_policy.out.get_weights()

    min_out = min(old_k.shape[0], new_k.shape[0])
    min_in = min(old_k.shape[1], new_k.shape[1])

    new_k[:min_out, :min_in] = old_k[:min_out, :min_in]
    new_b[:min_out] = old_b[:min_out]

    new_policy.out.set_weights([new_k, new_b])

    print(f"✅ Poids transférés ({min_u} neurones communs sur {new_recurrent.shape[0]})")
