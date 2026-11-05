from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Tuple

import chromobius
import numpy as np
import stim

class Chromobius:
    """
    Chromobius baseline for google color code d3 and d5 datasets.
    """

    @staticmethod
    def _prepare_decode(
        stim_file: str | Path, shot_indices: np.ndarray
    ) -> tuple[Any, np.ndarray]:
        """
        Compile a Chromobius decoder and prepare its detector data.

        :param stim_file: Path to the Stim circuit for the selected shots.
        :param shot_indices: Zero-based row indices in the detector-event file.
        :return: Compiled decoder and contiguous bit-packed detector events.
        """
        stim_file = Path(str(stim_file).replace("\\", "/"))
        circuit = stim.Circuit.from_file(stim_file)
        if circuit.num_observables != 1:
            raise ValueError(f"Expected one observable in {stim_file}.")

        bytes_per_shot = (circuit.num_detectors + 7) // 8
        if bytes_per_shot == 0:
            raise ValueError(f"No detectors in {stim_file}.")

        events_file = stim_file.with_name("detection_events.b8")
        events = np.memmap(events_file, mode="r", dtype=np.uint8)
        if events.size % bytes_per_shot:
            raise ValueError(f"Invalid .b8 file size: {events_file}.")
        events = events.reshape(-1, bytes_per_shot)

        if np.any(shot_indices < 0) or np.any(shot_indices >= len(events)):
            raise IndexError(f"Shot index outside {events_file}.")

        packed_events = np.ascontiguousarray(events[shot_indices])

        dem = circuit.detector_error_model(
            decompose_errors=False,
            approximate_disjoint_errors=True,
        )
        decoder = chromobius.compile_decoder_for_dem(dem)
        return decoder, packed_events

    @staticmethod
    def predict(dataloader: "CroissantDataLoader") -> Tuple[np.ndarray, float]:
        """
        Predict test-set logical flips using the Chromobius baseline.

        Only the batched decoder calls are timed. Dataset grouping, file I/O,
        decoder compilation, and prediction processing are excluded.

        :param dataloader: Data loader for the Croissant dataset.
        :return: Tuple of predicted logical flips and decoding time in seconds.
        """
        test_df = dataloader.get_test_df()
        required = {"stim_file", "source_shot_index"}
        missing = required.difference(test_df.columns)
        if missing:
            raise ValueError(f"Missing test.parquet columns: {sorted(missing)}")
        if test_df[["stim_file", "source_shot_index"]].isna().any().any():
            raise ValueError("Test shots contain missing Stim paths or shot indices.")

        predictions = np.empty(len(test_df), dtype=np.uint8)
        source_indices = test_df["source_shot_index"].to_numpy()
        decode_time = 0

        for stim_file, positions in test_df.groupby("stim_file", sort=False).indices.items():
            raw_indices = source_indices[positions]
            shot_indices = raw_indices.astype(np.int64)
            if not np.array_equal(raw_indices, shot_indices):
                raise ValueError(f"Non-integer shot indices for {stim_file}.")

            decoder, packed_events = Chromobius._prepare_decode(
                stim_file, shot_indices
            )

            # ONLY the compiled native decoder call is measured.
            start_time = time.perf_counter()
            packed_predictions = decoder.predict_obs_flips_from_dets_bit_packed(
                packed_events
            )
            decode_time += time.perf_counter() - start_time

            predictions[positions] = packed_predictions[:, 0] & 1

        return predictions, decode_time
