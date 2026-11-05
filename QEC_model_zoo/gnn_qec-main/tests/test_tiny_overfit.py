import unittest

import numpy as np

try:
    import tensorflow as tf

    from qec_gnn.model import build_gnn_model

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow is not installed")
class TinyOverfitTest(unittest.TestCase):
    def test_model_can_fit_simple_graph_rule(self):
        tf.keras.utils.set_random_seed(11)
        x = np.zeros((16, 4, 4), dtype=np.float32)
        a = np.tile(np.eye(4, dtype=np.float32), (16, 1, 1))
        mask = np.ones((16, 4), dtype=np.float32)
        y = np.zeros((16,), dtype=np.float32)

        for i in range(16):
            label = i % 2
            x[i, :, 0] = float(label)
            y[i] = float(label)

        model = build_gnn_model(max_nodes=4, feature_dim=4, hidden_dim=16, dropout=0.0, learning_rate=0.01)
        history = model.fit([x, a, mask], y, epochs=40, batch_size=4, verbose=0)

        self.assertGreaterEqual(history.history["accuracy"][-1], 0.95)


if __name__ == "__main__":
    unittest.main()
