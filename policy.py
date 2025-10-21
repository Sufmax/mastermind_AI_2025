# policy.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Masking
import config

class Policy(tf.Module):
    """
    Policy TF2 batch-ready.
    Input: history tensor (batch, time, 6), mask (batch, time) boolean/int
    Output: probs (batch, 5, 4) in [0,1]
    """

    def __init__(self, lstm_hidden_size=None, name=None):
        super().__init__(name=name)
        hidden_size = lstm_hidden_size or config.lstm_hidden_size
        self.masking = Masking(mask_value=0.0)
        self.lstm = LSTM(hidden_size, return_state=False, return_sequences=False)
        self.out = Dense(5 * 4)

    @property
    def variables(self):
        return self.masking.trainable_variables + \
               self.lstm.trainable_variables + \
               self.out.trainable_variables

    def __call__(self, history, mask=None, with_sigmoid=True):
        # history: (batch, time, features) or (time,features) (we convert)
        h = tf.convert_to_tensor(history, dtype=tf.float32)
        if len(h.shape) == 2:
            h = tf.expand_dims(h, axis=0)  # (1, time, features)

        # apply masking layer (it will just pass data but keep shapes)
        h = self.masking(h)

        # prepare mask for LSTM: boolean (batch, time)
        if mask is not None:
            mask_t = tf.cast(mask, tf.bool)
            lstm_out = self.lstm(h, mask=mask_t)
        else:
            lstm_out = self.lstm(h)

        logits = self.out(lstm_out)                 # (batch, 20)
        logits = tf.reshape(logits, (-1, 5, 4))     # (batch, 5, 4)

        if with_sigmoid:
            return tf.sigmoid(logits)               # probabilities
        return logits
