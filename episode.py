# episode.py
import config
import numpy as np
import tensorflow as tf

class Episode:
    """
    Génère un épisode en utilisant la Policy TF2 adaptée.
    L'historique interne est un tableau (max_episode_length, 6) et un mask (max_episode_length,)
    Chaque ligne : [r1,r2,r3,r4, score_color_norm, score_place_norm]
    """

    def __init__(self, policy, secret):
        """
        secret: int (ancien encodage), or string '0123', or list of 4 ints
        """
        self.policy = policy
        self.max_len = config.max_episode_length
        self._secret_digits = self._normalize_secret(secret)  # list of 4 ints in 0..5

    def _normalize_secret(self, secret):
        if isinstance(secret, int):
            # decode base-6 integer -> 4-digit string
            assert 0 <= secret < config.max_guesses
            s = []
            idx = secret
            for i in range(4):
                s.append(str(idx % 6))
                idx //= 6
            s = "".join(reversed(s)).zfill(4)
            return [int(ch) for ch in s]
        if isinstance(secret, str):
            assert len(secret) == 4
            return [int(c) for c in secret]
        if isinstance(secret, (list, tuple, np.ndarray)):
            assert len(secret) == 4
            return [int(x) for x in secret]
        raise ValueError("Secret format not supported")

    @staticmethod
    def digits_to_index(digits):
        index = 0
        for i, d in enumerate(digits):
            index += int(d) * (6 ** (3 - i))
        return index

    @staticmethod
    def index_to_digits(index):
        digits = []
        for i in range(4):
            power = 6 ** (3 - i)
            digits.append((index // power) % 6)
        return digits

    @staticmethod
    def compute_feedback(secret_digits, guess_digits):
        """
        Retourne (score_color_norm, score_place_norm) et aussi discrete plus/minus si nécessaire.
        score_place = exact matches / 4
        score_color = total color matches / 4 (including place matches)
        """
        # place matches
        place = sum(int(s == g) for s, g in zip(secret_digits, guess_digits))
        # color matches (multiset intersection)
        secret_counts = [secret_digits.count(i) for i in range(6)]
        guess_counts = [guess_digits.count(i) for i in range(6)]
        common = sum(min(secret_counts[i], guess_counts[i]) for i in range(6))
        color = common
        # normalized
        return color / 4.0, place / 4.0, (color, place)

    @staticmethod
    def _sample_binary_matrix(probs):
        """
        probs: numpy array shape (5,4) with values in [0,1]
        return: binary matrix shape (5,4) sampled from Bernoulli
        and log_prob of that sample under independent Bernoullis
        """
        # sample bernoulli
        bern = np.random.binomial(1, probs)
        # compute log_prob = sum s*log(p) + (1-s)*log(1-p)
        eps = 1e-9
        p = np.clip(probs, eps, 1 - eps)
        log_prob_matrix = bern * np.log(p) + (1 - bern) * np.log(1 - p)
        log_prob = float(np.sum(log_prob_matrix))
        return bern, log_prob

    @staticmethod
    def binary_matrix_to_guess_digits(binary):
        """
        binary: shape (5,4) values 0/1
        sum across rows -> per-position counts 0..5
        returns list of 4 ints in 0..5
        """
        counts = np.sum(binary, axis=0)  # shape (4,)
        digits = [int(int(c)) for c in counts.tolist()]
        return digits

    def generate(self):
        """
        Génère et renvoie une liste d'étapes. Chaque étape est dict:
        {
            'guess_digits': [4 ints],
            'guess_index': int,
            'feedback_color': float (0..1),
            'feedback_place': float (0..1),
            'log_prob': float
        }
        La première entrée est un 'start' token (zeros in history).
        """
        history_rows = []  # will be list of (r1..r4, score_color, score_place)
        steps = []

        # initial start token (all zeros) as per spec
        for t in range(self.max_len):
            # build input for call: shape (max_len,6) with zeros & mask
            mask = np.array([1 if i < len(history_rows) else 0 for i in range(self.max_len)], dtype=np.int32)
            # build history matrix
            hist = np.zeros((self.max_len, 6), dtype=np.float32)
            if len(history_rows) > 0:
                hist[:len(history_rows), :] = np.array(history_rows, dtype=np.float32)

            # call policy -> probs shape (5,4)
            probs = self.policy(hist, mask=mask, with_sigmoid=True).numpy()

            # sample binary matrix and log_prob
            binary, log_prob = self._sample_binary_matrix(probs)

            # convert binary -> guess digits
            guess_digits = self.binary_matrix_to_guess_digits(binary)
            guess_index = self.digits_to_index(guess_digits)

            # compute feedback
            color_norm, place_norm, (color_raw, place_raw) = self.compute_feedback(self._secret_digits, guess_digits)

            step = {
                'guess_digits': guess_digits,
                'guess_index': guess_index,
                'feedback_color': color_norm,
                'feedback_place': place_norm,
                'log_prob': log_prob
            }
            steps.append(step)

            # append to history_rows the normalized row for this turn
            row = [d / 5.0 for d in guess_digits] + [color_norm, place_norm]
            history_rows.append(row)

            # stop if correct
            if place_raw == 4:
                break

            # if reached max_len loop will end next iteration automatically
            if len(history_rows) >= self.max_len:
                break

        return steps
