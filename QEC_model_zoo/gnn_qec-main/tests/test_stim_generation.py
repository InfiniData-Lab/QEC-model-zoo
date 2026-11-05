import unittest

import numpy as np

from qec_gnn.stim_utils import get_detector_coord_array, make_surface_code_circuit


class StimGenerationTest(unittest.TestCase):
    def test_surface_code_samples_have_expected_shapes(self):
        circuit = make_surface_code_circuit(distance=3, rounds=3, p=0.01)
        sampler = circuit.compile_detector_sampler(seed=7)
        detectors, observables = sampler.sample(shots=8, separate_observables=True)

        self.assertEqual(detectors.shape, (8, circuit.num_detectors))
        self.assertEqual(observables.shape, (8, circuit.num_observables))
        self.assertEqual(observables.shape[1], 1)

    def test_detector_coordinates_are_normalized_xyz(self):
        circuit = make_surface_code_circuit(distance=3, rounds=3, p=0.01)
        coords = get_detector_coord_array(circuit)

        self.assertEqual(coords.shape, (circuit.num_detectors, 3))
        self.assertEqual(coords.dtype, np.float32)
        self.assertTrue(np.all(coords >= 0.0))
        self.assertTrue(np.all(coords <= 1.0))


if __name__ == "__main__":
    unittest.main()
