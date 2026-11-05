### NVIDIA pre-decoder benchmarks for Color code

This implementation is derived from the NVIDIA Ising-Decoding project: https://github.com/NVIDIA/Ising-Decoding. 

Pre-decoder benchmarks on the triangular color code with the superdense memory circuit:

- **Chromobius baseline:** logical error rate without a pre-decoder, swept over distance and physical error rate. Runs on CPU.
- **NVIDIA Model B:** the pretrained cascade pre-decoder (`nvidia/Ising-Decoder-ColorCode-1-Fast`, trained at d=13, 13 rounds), followed by Chromobius, compared with Chromobius alone.

The Python code lives in the shared engine `../../utils/predecoder/` (see its README). The color-specific parts there are `qec/color_code/`, `data/datapipe_stim_color.py`, `evaluation/logical_error_rate_color.py` and `evaluation/color_baseline_benchmark.py`. This folder holds only launchers, configs and outputs.

All commands below start from the repository root. The folder name `color code` contains a space, so always quote the path.

#### layout

| Path | Contents |
| --- | --- |
| `QEC-PD-COLOR-baseline.sh` | Chromobius baseline sweep (CPU) |
| `QEC-PD-COLOR-ModelB-cpu.sh` | Model B inference on CPU (fp32), d=5 by default |
| `QEC-PD-COLOR-ModelB-gpu.sh` | Model B inference on GPU (fp16), d=13 and 262144 shots by default |
| `configs/config_color_inference_modelB.yaml` | Model B config (4 input/output channels) |
| `outputs/` | One folder per experiment: `logs/`, `config_used/`, result JSON |

#### setup

1. Activate an environment with the pre-decoder dependencies, including `chromobius` (locally: `conda activate Ising-decoding`). Note that "Ising-decoding" is a conda environment for experiments. We created an environment named Ising-decoding and installed all necessary dependencies by following the setup of the Dependencies of the README of NVIDIA Ising-Decoding GitHub repo project.  For detailed enviroment configuration, please go through the Dependencies of the README of NVIDIA Ising-Decoding GitHub repo project. It also reported troubleshooting. 
2. For Model B, download the gated NVIDIA checkpoint. The launchers look for it in `QEC_model_zoo/model/predecoders/`: 

   ```bash
   hf download nvidia/Ising-Decoder-ColorCode-1-Fast \
       --local-dir QEC_model_zoo/model/predecoders/Ising-Decoder-ColorCode-1-Fast
   ```
For detailed information of getting pre-trained models, please go through the Inference (pre-trained models) of the README of NVIDIA Ising-Decoding GitHub repo project. 

#### how to run Model B on CPU (macOS, Linux)?

```bash
cd "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder"

# Defaults: 6 runs, shots per basis per run
bash QEC-PD-COLOR-ModelB-cpu.sh
```

It reports the logical error rate for PD+Chromobius and for Chromobius alone, the detector-density reduction and the decoder speedup. Like the Surface launcher, each run is saved directly under `outputs/bench_color_modelB_cpu_d5_runN/`, containing its own `logs/` and `config_used/`. Re-running the same experiment replaces each selected run’s console log. Use a different `EXPERIMENT_NAME` to retain separate experiments. `outputs/bench_color_modelB_cpu_d5_summary/` contains `runs.csv`, `summary.json` and `summary.txt`, with means and sample standard deviations across runs, grouped by basis. LER is per round; speedup measures Chromobius decoding alone, excluding neural inference. Undefined LER improvement ratios are omitted from averages, with the included count reported as `n`. Keep the distance small on CPU: at d=13 each basis takes minutes.

The CPU launcher accepts the same sample-count variable as Surface: `PREDECODER_INFERENCE_NUM_SAMPLES`. Precedence is `PREDECODER_INFERENCE_NUM_SAMPLES` > `TEST_SAMPLES` > `SHOTS` > default `20000` (empty values use the next fallback). The resolved value is passed to the runner as `TEST_SAMPLES`, then to Color evaluation as `PREDECODER_TEST_SAMPLES`. Each run uses fresh unseeded Monte Carlo sampling. `PREDECODER_TEST_MEAS_BASIS=both` evaluates X and Z separately, each with the requested number of shots. The launcher reads its YAML directly without replacing the shared GPU config.

#### how to run Model B on GPU? (GPU server)

```bash
cd "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder"
bash QEC-PD-COLOR-ModelB-gpu.sh
CUDA_VISIBLE_DEVICES=1 PREDECODER_TEST_P_ERROR=0.001 bash QEC-PD-COLOR-ModelB-gpu.sh
```

#### settings (Model B launchers)

| Variable | CPU default | GPU default |
| --- | --- | --- |
| `DISTANCE`, `N_ROUNDS` | `5`, `5` | `13`, `13` |
| `TEST_SAMPLES`, `TEST_BATCH_SIZE` | `20000`, `400` | `262144`, `2048` |
| `RUNS` | `6` | not used |
| `PREDECODER_TEST_P_ERROR` | `0.003` | `0.002` |
| `PREDECODER_TEST_MEAS_BASIS` | `Z` | `both` |
| `EXPERIMENT_NAME` | `bench_color_modelB_cpu_d5` | `bench_color_modelB_gpu_d13` |
| `RESULTS_BASE_DIR` | `outputs/` in this folder | `outputs/` in this folder |
| `PREDECODER_SAFETENSORS_CHECKPOINT` | checkpoint under `QEC_model_zoo/model/predecoders/` | same |

#### how to run the Chromobius baseline?

```bash
cd "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder"

# Quick check
DISTANCES="3" P_VALUES="0.004" SHOTS=2000 bash QEC-PD-COLOR-baseline.sh

# Defaults: d = 3 5 7, p = 0.004 0.008, 20000 shots
bash QEC-PD-COLOR-baseline.sh
```

#### result records
Results are written to `outputs/bench_color_baseline/color_baseline.json`; change the folder with `EXPERIMENT_NAME`.

Given the outputs, the benchmark results can be collected via: 

```bash 
python QEC_model_zoo/benchmark/utils/predecoder/analysis/extract_color_results.py
```
The results are saved to `benchmark/color code/NVIDIA-predecoder/outputs/bench_color_modelB_cpu_d5_results.csv`. 