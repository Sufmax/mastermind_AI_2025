import config
import numpy as np
import tensorflow as tf

class Episode:
    def __init__(self, policy, secret):
        self.policy = policy
        self.max_len = config.max_episode_length
        self._secret_digits = self._normalize_secret(secret)

    def _normalize_secret(self, secret):
        if isinstance(secret, int):
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
        place = sum(int(s == g) for s, g in zip(secret_digits, guess_digits))
        secret_counts = [secret_digits.count(i) for i in range(6)]
        guess_counts = [guess_digits.count(i) for i in range(6)]
        common = sum(min(secret_counts[i], guess_counts[i]) for i in range(6))
        color = common
        return color / 4.0, place / 4.0, (color, place)

    @staticmethod
    def _sample_binary_matrix(probs):
        bern = np.random.binomial(1, probs)
        eps = 1e-9
        p = np.clip(probs, eps, 1 - eps)
        log_prob_matrix = bern * np.log(p) + (1 - bern) * np.log(1 - p)
        log_prob = float(np.sum(log_prob_matrix))
        return bern, log_prob

    @staticmethod
    def binary_matrix_to_guess_digits(binary):
        counts = np.sum(binary, axis=0)
        digits = [int(int(c)) for c in counts.tolist()]
        return digits

    def generate(self):
        history_rows = []
        steps = []
        for t in range(self.max_len):
            mask = np.array([1 if i < len(history_rows) else 0 for i in range(self.max_len)], dtype=np.int32)
            hist = np.zeros((self.max_len, 6), dtype=np.float32)
            if len(history_rows) > 0:
                hist[:len(history_rows), :] = np.array(history_rows, dtype=np.float32)
            probs = self.policy(hist, mask=mask, with_sigmoid=True).numpy()
            binary, log_prob = self._sample_binary_matrix(probs)
            guess_digits = self.binary_matrix_to_guess_digits(binary)
            guess_index = self.digits_to_index(guess_digits)
            color_norm, place_norm, (color_raw, place_raw) = self.compute_feedback(self._secret_digits, guess_digits)
            step = {
                'guess_digits': guess_digits,
                'guess_index': guess_index,
                'feedback_color': color_norm,
                'feedback_place': place_norm,
                'log_prob': log_prob
            }
            steps.append(step)
            row = [d / 5.0 for d in guess_digits] + [color_norm, place_norm]
            history_rows.append(row)
            if place_raw == 4:
                break
            if len(history_rows) >= self.max_len:
                break
        return steps
