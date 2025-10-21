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
        # create a fake history: two turns
        hist = np.zeros((30,6), dtype=np.float32)
        hist[0,:] = [0/5,1/5,2/5,3/5, 0.25, 0.25]
        hist[1,:] = [1/5,2/5,3/5,4/5, 0.5, 0.25]
        mask = np.array([1,1] + [0]*28, dtype=np.int32)

        expected = pol(hist, mask=mask).numpy()

        with tempfile.TemporaryDirectory() as tdir:
            path = os.path.join(tdir, "ckpt")
            ckpt = tf.train.Checkpoint(model=pol)
            ckpt.write(path)

            pol2 = Policy()
            diff_before = np.linalg.norm(pol2(hist, mask=mask).numpy() - expected)
            self.assertGreater(diff_before, 1e-6)

            ckpt2 = tf.train.Checkpoint(model=pol2)
            ckpt2.read(path).assert_existing_objects_matched()

            diff_after = np.linalg.norm(pol2(hist, mask=mask).numpy() - expected)
            self.assertLessEqual(diff_after, 1e-5)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
