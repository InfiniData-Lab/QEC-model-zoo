import unittest

import numpy as np

from qec_gnn.graph_builder import build_graph_dataset, build_graph_from_shot, build_knn_adjacency, normalize_adjacency


class GraphBuilderTest(unittest.TestCase):
    def test_knn_adjacency_is_symmetric(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        adj = build_knn_adjacency(coords, k=1)

        self.assertEqual(adj.shape, (3, 3))
        self.assertTrue(np.allclose(adj, adj.T))
        self.assertTrue(np.all(np.diag(adj) == 1.0))

    def test_empty_shot_gets_dummy_node(self):
        sample = np.zeros((5,), dtype=np.uint8)
        coords = np.zeros((5, 3), dtype=np.float32)

        x, a, mask, node_count, truncated = build_graph_from_shot(sample, coords, max_nodes=4, k=2)

        self.assertEqual(x.shape, (4, 4))
        self.assertEqual(a.shape, (4, 4))
        self.assertEqual(mask.shape, (4,))
        self.assertEqual(node_count, 1)
        self.assertFalse(truncated)
        self.assertEqual(mask.sum(), 1.0)

    def test_graph_dataset_preserves_labels(self):
        detectors = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.uint8)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        graph_data = build_graph_dataset(detectors, labels, coords, max_nodes=4, k=1)

        self.assertEqual(graph_data["X"].shape, (2, 4, 4))
        self.assertEqual(graph_data["A"].shape, (2, 4, 4))
        self.assertTrue(np.array_equal(graph_data["y"], labels))

    def test_normalized_adjacency_is_finite(self):
        adj = np.eye(3, dtype=np.float32)
        normalized = normalize_adjacency(adj)

        self.assertEqual(normalized.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(normalized)))


if __name__ == "__main__":
    unittest.main()
