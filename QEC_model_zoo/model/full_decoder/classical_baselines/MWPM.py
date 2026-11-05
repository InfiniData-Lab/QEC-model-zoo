"""MWPM baseline using source_shot_id from the annotated shot dataset."""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pymatching
import stim


class MWPM:
    """Decode Google QEC shots in dataframe row order."""

    @staticmethod
    def predict(df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """Return one logical-flip prediction per input row and decode time.

        Preparation (file loading, decoder construction, shot selection) is
        excluded from timing. source_shot_id indexes each experiment's own
        detection_events.b8 file. Relative paths resolve from the working
        directory. Missing mappings raise so predictions stay aligned with
        labels. The baseline expects one logical observable per circuit.
        """
        required = ['stim_file']
        # Either use shot_id or the source_shot_id column (latter is a defined mapping if shot_ids do not match exactly between raw data and processed tabular data)
        source_shot_id_col = 'source_shot_id' if 'source_shot_id' in df.columns else 'shot_id'
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f'Missing columns: {missing}. Regenerate the annotated parquet '
                'with the updated notebook; shot_id may be global and cannot '
                'directly index an experiment-local .b8 file.'
            )
        if df[required].isna().any().any():
            raise ValueError('Unresolved Stim paths/local shot IDs; predictions must remain aligned with all input rows.')
        if df.empty:
            return np.empty(0, dtype=np.uint8), 0.0

        frame = df.reset_index(drop=True)
        folders = frame['stim_file'].map(
            lambda p: Path(str(p).replace('\\', '/')).parent
        )
        batches = []
        for folder, row_idx in folders.groupby(folders, sort=False).groups.items():
            row_idx = np.asarray(row_idx, dtype=np.int64)
            dem = stim.DetectorErrorModel.from_file(
                folder / 'circuit_detector_error_model.dem'
            )
            if dem.num_observables != 1:
                raise ValueError(f'{folder}: expected one logical observable; found {dem.num_observables}.')
            matching = pymatching.Matching.from_detector_error_model(dem)
            all_dets = stim.read_shot_data_file(
                path=folder / 'detection_events.b8',
                format='b8',
                num_detectors=dem.num_detectors,
                bit_packed=True,
            )
            ids = pd.to_numeric(frame.iloc[row_idx][source_shot_id_col], errors='raise')
            local_ids = ids.to_numpy(dtype=np.int64)
            if ids.ne(local_ids).any():
                raise ValueError(f'{folder}: source_shot_id must contain integers.')
            invalid = (local_ids < 0) | (local_ids >= len(all_dets))
            if invalid.any():
                raise ValueError(
                    f'{folder}: source_shot_id {local_ids[invalid][0]} is outside '
                    f'0..{len(all_dets) - 1}. Check the raw-to-source mapping.'
                )
            if 'source_num_shots' in frame.columns:
                counts = frame.iloc[row_idx]['source_num_shots']
                if counts.isna().any() or counts.ne(len(all_dets)).any():
                    raise ValueError(f'{folder}: detection_events.b8 length disagrees with the manifest.')
            batches.append((row_idx, matching, all_dets[local_ids]))

        predicted = np.empty(len(frame), dtype=np.uint8)
        start = time.perf_counter()
        for row_idx, matching, dets in batches:
            predicted[row_idx] = matching.decode_batch(
                dets, bit_packed_shots=True,
            )[:, 0]
        elapsed = time.perf_counter() - start
        print(f'Prediction time: {elapsed:.6f} s')
        if elapsed > 0:
            print(f'Throughput: {len(frame) / elapsed:,.0f} shots/s')
        return predicted, elapsed
