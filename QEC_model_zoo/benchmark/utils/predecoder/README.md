### NVIDIA pre-decoder engine (shared)

Shared code for the NVIDIA Ising-Decoding pre-decoders, used by the surface code and color code benchmarks:

- `../../surface code/NVIDIA-predecoder/`
- `../../color code/NVIDIA-predecoder/`

Original source: NVIDIA Ising-Decoding (Apache-2.0), extended with the color-code pipeline (Model B, Chromobius baseline) and custom tabular pre-decoders (MLP, LSTM, TabM, TabR).

This folder is the Python import root. Scripts set `PYTHONPATH` to it and `cd` into it before running, so imports such as `from qec.color_code import ...` and `from model.factory import ModelFactory` resolve here. It must not share a `PYTHONPATH` with `QEC_model_zoo/` (the `main.py` full-decoder pipeline), because both define top-level `model`, `data`, `utils` and `analysis` packages.

#### layout

| Folder | Contents |
| --- | --- |
| `workflows/` | Hydra entry points: `run.py` (official models), `run_custom_models.py` (custom models, color Model B), `config_validator.py` |
| `conf/` | Public and internal configs (`config_public.yaml` and others). Code-family configs live in each family's `configs/` folder |
| `qec/` | Code definitions and circuits: `surface_code/`, `color_code/`, noise model, DEM sampling |
| `data/` | Stim datapipes (surface and color), data factory, frame precomputation |
| `model/` | Pre-decoder models (`predecoder.py`), custom tensor models (`tensor_models.py`), factory and registry |
| `training/` | Training loop, distributed setup, precision, logging |
| `evaluation/` | Inference, logical error rate (surface and color), Chromobius baseline, metrics |
| `analysis/` | `predecoder_log_stats.py`: aggregates inference logs into mean/std tables |
| `utils/export/` | Checkpoint to SafeTensors export |
| `scripts/custom/` | Runners used by the family launchers: `run_custom_inference.sh`, `run_custom_train.sh`, `prepare_custom_checkpoint4inference.sh` |
| `scripts/nvidia/` | Upstream NVIDIA runners (`local_run.sh` is GPU-only), cluster and CI scripts |
| `tests/` | Unit tests |
| `frames_data/` | Precomputed DEM frames |

Pretrained NVIDIA checkpoints are not stored here. They stay in `QEC_model_zoo/model/predecoders/` (gitignored), which is where the family launchers look for them.

#### how to run the unit tests?

With the pre-decoder environment active (locally: conda env `Ising-decoding`, Python 3.11):

```bash
cd QEC_model_zoo/benchmark/utils/predecoder
PYTHONPATH="$PWD" python -m unittest discover -s tests -p "test_*.py"
```

On a CPU-only Mac this runs 238 tests in about 10 s. 46 are skipped (GPU-only) and 9 fail for reasons that predate the move. For example, the requirements files that `test_onnx_quant_workflow` and `test_tensorrt_fallback` expect are not in this repo, and the public model files for `test_inference_public_model` are missing.

#### outputs

`scripts/custom/run_custom_*.sh` write logs, `config_used/` and model checkpoints to `RESULTS_BASE_DIR/EXPERIMENT_NAME`. The family launchers set `RESULTS_BASE_DIR` to their own `outputs/` folder. When you call the runners directly, pass an absolute path; relative paths resolve from this folder.

`run_custom_*.sh` temporarily copy `CUSTOM_CONFIG` over `conf/config_public.yaml` and restore it on exit, so do not run two of them at the same time.

Logs contain only relative file paths. The runners, `local_run.sh` and `QEC-PD-COLOR-baseline.sh` pass their output through `relpath_filter` from `scripts/custom/log_paths.sh`. It rewrites absolute paths under the repository root to repo-relative paths (e.g. `QEC_model_zoo/benchmark/...`) and other paths under `$HOME` to `~/...`. After a run, `relpath_logs` applies the same rewrite to the `*.log` files Python wrote in the output folder (e.g. the Hydra job log). Source `log_paths.sh` in any new launcher that writes logs.
