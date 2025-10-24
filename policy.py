import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Masking, Dropout
import config

class Policy(tf.keras.Model):
    """
    Policy TF2 batch-ready.
    Input: tuple (history, mask)
    Output: probs (batch, 5, 4) in [0,1]
    """

    def __init__(self, lstm_hidden_size=None, name='Policy'):
        super().__init__(name=name)
        hidden_size = lstm_hidden_size or config.lstm_hidden_size
        self.masking = Masking(mask_value=0.0)
        # LSTM principal
        self.lstm = LSTM(hidden_size, return_state=False, return_sequences=False, implementation=2)
        # Plusieurs couches Dense pour plus de "puissance"
        self.dense1 = Dense(512, activation='relu')
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(256, activation='relu')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(128, activation='relu')
        self.out = Dense(5 * 4, name="output_dense")

    def call(self, inputs, with_sigmoid=True, training=False):
        history, mask = inputs
        h = tf.convert_to_tensor(history, dtype=tf.float32)
        if len(h.shape) == 2:
            h = tf.expand_dims(h, axis=0)
        # Masque right-padded booléen
        if mask is not None:
            mask_t = tf.cast(mask, tf.bool)
            if len(mask_t.shape) == 1:
                mask_t = tf.expand_dims(mask_t, axis=0)
            lstm_out = self.lstm(h, mask=mask_t, training=training)
        else:
            lstm_out = self.lstm(h, training=training)
        # Empilement de couches Dense
        x = self.dense1(lstm_out)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        logits = self.out(x)
        logits = tf.reshape(logits, (-1, 5, 4))
        if with_sigmoid:
            return tf.sigmoid(logits)
        return logits
