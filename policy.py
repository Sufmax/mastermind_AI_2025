import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
import itertools
import config

class Policy(tf.Module):
    """
    Policy for Mastermind using TensorFlow 2.
    """

    def __init__(self):
        super().__init__()
        self.guess_embedding = Embedding(config.max_guesses + 1,
                                         config.guess_embedding_size)
        self.feedback_embedding = Embedding(config.max_feedback + 1,
                                            config.feedback_embedding_size)
        self.lstm = LSTM(config.lstm_hidden_size, return_state=True, return_sequences=True)
        self.dense = Dense(config.max_guesses)

    @property
    def variables(self):
        return self.guess_embedding.trainable_variables + \
               self.feedback_embedding.trainable_variables + \
               self.lstm.trainable_variables + \
               self.dense.trainable_variables

    def __call__(self, game_state, with_softmax=True):
        """
        Forward pass to get action logits
        """
        seq = []
        for guess, feedback in game_state:
            guess_tensor = tf.convert_to_tensor([guess], dtype=tf.int32)
            feedback_tensor = tf.convert_to_tensor([feedback], dtype=tf.int32)
            guess_emb = self.guess_embedding(guess_tensor)
            feedback_emb = self.feedback_embedding(feedback_tensor)
            combined = tf.concat([guess_emb, feedback_emb], axis=-1)
            seq.append(combined)

        x = tf.stack(seq, axis=1)  # shape (batch=1, time, features)
        lstm_out, h, c = self.lstm(x)
        logits = self.dense(lstm_out[:, -1, :])  # last time step
        if with_softmax:
            return tf.nn.softmax(logits)
        return logits
