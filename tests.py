import unittest
import tensorflow as tf
import tempfile
import os
import numpy as np
from policy import Policy

tf.random.set_seed(230)

class TestCheckpoints(unittest.TestCase):

    def test_save_restore(self):
        pol = Policy()
        episode = [(0, 0), (1, 0), (2, 3)]
        expected = pol(episode).numpy()

        with tempfile.TemporaryDirectory() as tdir:
            path = os.path.join(tdir, "checkpt")

            # sauvegarde avec tf.train.Checkpoint
            ckpt = tf.train.Checkpoint(model=pol)
            ckpt.write(path)

            # créer un nouveau modèle
            pol2 = Policy()

            # vérifier qu'il est différent avant restore
            diff_before = np.linalg.norm(pol2(episode).numpy() - expected)
            self.assertGreater(diff_before, 0.0001)

            # restaurer le checkpoint
            ckpt2 = tf.train.Checkpoint(model=pol2)
            ckpt2.read(path).assert_existing_objects_matched()

            # vérifier que la sortie est maintenant identique
            diff_after = np.linalg.norm(pol2(episode).numpy() - expected)
            self.assertLessEqual(diff_after, 1e-5)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
