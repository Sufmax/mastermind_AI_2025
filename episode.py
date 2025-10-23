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
            for _ in range(4):
                s.append(str(idx % 6))
                idx //= 6
            return [int(ch) for ch in "".join(reversed(s)).zfill(4)]
        if isinstance(secret, str):
            assert len(secret) == 4 and all(c in "012345" for c in secret)
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
    def compute_feedback(secret_digits, guess_digits):
        place = sum(int(s == g) for s, g in zip(secret_digits, guess_digits))
        secret_counts = [secret_digits.count(i) for i in range(6)]
        guess_counts = [guess_digits.count(i) for i in range(6)]
        common = sum(min(sc, gc) for sc, gc in zip(secret_counts, guess_counts))
        return common / 4.0, place / 4.0, (common, place)

    @staticmethod
    def _sample_binary_matrix(probs):
        bern = np.random.binomial(1, probs)
        eps = 1e-9
        p = np.clip(probs, eps, 1 - eps)
        log_prob_matrix = bern * np.log(p) + (1 - bern) * np.log(1 - p)
        return bern, float(np.sum(log_prob_matrix))

    @staticmethod
    def binary_matrix_to_guess_digits(binary):
        if binary.ndim == 3:
            binary = np.squeeze(binary, axis=0)
        counts = np.sum(binary, axis=0)
        return [int(round(c)) for c in counts]

    def generate(self):
        history_rows = []
        steps = []
        for _ in range(self.max_len):
            hist = np.zeros((self.max_len, 6), dtype=np.float32)
            if history_rows:
                hist[:len(history_rows), :] = np.array(history_rows, dtype=np.float32)
            mask = np.zeros((self.max_len,), dtype=np.int32)
            mask[:len(history_rows)] = 1
            hist_batch = np.expand_dims(hist, axis=0)
            mask_batch = np.expand_dims(mask, axis=0)
            probs = self.policy(hist_batch, mask=mask_batch, with_sigmoid=True).numpy()
            binary, log_prob = self._sample_binary_matrix(probs)
            guess_digits = self.binary_matrix_to_guess_digits(binary)
            color_norm, place_norm, (color_raw, place_raw) = self.compute_feedback(self._secret_digits, guess_digits)
            steps.append({
                'guess_digits': guess_digits,
                'guess_index': self.digits_to_index(guess_digits),
                'feedback_color': color_norm,
                'feedback_place': place_norm,
                'log_prob': log_prob
            })
            row = [d / 5.0 for d in guess_digits] + [color_norm, place_norm]
            history_rows.append(row)
            if place_raw == 4:
                break
        return steps
