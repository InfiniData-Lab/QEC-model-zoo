### NVIDIA pre-decoder benchmarks for Surface code

This implementation is derived from the NVIDIA Ising-Decoding project: https://github.com/NVIDIA/Ising-Decoding.  

Pre-decoder benchmarks on the rotated surface code (e.g., d=9, 9 rounds, circuit-level noise around p=0.003):

- **NVIDIA Model 1 / Fast:** the pretrained 3D CNN pre-decoder (`nvidia/Ising-Decoder-SurfaceCode-1-Fast`, receptive field R=9), followed by PyMatching.
- **Custom pre-decoders:** MLP, LSTM, TabM and TabR (`custom_model_id` 90–93), trained and evaluated in the same pipeline. These are 
- **NVIDIA CNN training:** the official training run, which needs a GPU.

The Python code lives in the shared engine `../../utils/predecoder/` (see its README). This folder holds only launchers, configs and outputs.

All commands below start from the repository root. The folder name `surface code` contains a space, so always quote the path.

#### layout

| Path | Contents |
| --- | --- |
| `QEC-PD-SURFACE-Fast-cpu.sh` | Model 1 / Fast inference on CPU, repeated `RUNS` times and then aggregated (mean/std) |
| `QEC-PD-D9R9-{MLP,LSTM,TABM,TABR}.sh` | Custom pre-decoder inference; the training and checkpoint-prep steps are commented at the top |
| `QEC-PD-D9R9-CNN.sh` | Official CNN training + inference via `local_run.sh` (GPU only) |
| `QEC-PD-D9R9-CNN-F.sh`, `run_cpu_test.sh` | Older scripts that load `.pt` checkpoints not stored in this repo |
| `configs/` | `config_surface_inference_fast.yaml`, `config_2080ti_{train,inference}_{mlp,lstm,tabm,tabr}_p003.yaml` |
| `outputs/` | One folder per experiment: `logs/`, `config_used/`, `models/` |

#### setup

1. Activate an environment with the pre-decoder dependencies (locally: `conda activate Ising-decoding`). Note that "Ising-decoding" is a conda environment for experiments. We created an environment named Ising-decoding and installed all necessary dependencies by following the setup of the Dependencies of the README of NVIDIA Ising-Decoding GitHub repo project.  For detailed enviroment configuration, please go through the Dependencies of the README of NVIDIA Ising-Decoding GitHub repo project. It also reported troubleshooting. 
2. Download the gated NVIDIA checkpoint. The launchers look for it in `QEC_model_zoo/model/predecoders/`:

   ```bash
   hf download nvidia/Ising-Decoder-SurfaceCode-1-Fast \
       --local-dir QEC_model_zoo/model/predecoders/Ising-Decoder-SurfaceCode-1-Fast
   ```
For detailed information of getting pre-trained models, please go through the Inference (pre-trained models) of the README of NVIDIA Ising-Decoding GitHub repo project. 

#### how to run Model 1 / Fast on CPU (macOS, Linux)?

```bash
cd "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder"

# Quick check: 1 run, 256 shots per basis
RUNS=1 PREDECODER_INFERENCE_NUM_SAMPLES=256 PREDECODER_INFERENCE_LATENCY_SAMPLES=64 \
    bash QEC-PD-SURFACE-Fast-cpu.sh

# Defaults: 6 runs, 4096 shots per basis
bash QEC-PD-SURFACE-Fast-cpu.sh
```

- Each run reports the logical error rate before and after the pre-decoder for the X and Z bases, plus the PyMatching speedup. The script ends with a mean/std table across runs.
- Results go to `outputs/bench_surface_fast_cpu_run<i>/`. Change the prefix with `EXPERIMENT_NAME`.
- The quick check took about 50 s on an Apple M4. With 256 shots the logical error rate is too noisy to compare decoders; use it only to confirm the pipeline works.

#### how to run the custom pre-decoders? (GPU server)

```bash
cd "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder"
bash QEC-PD-D9R9-MLP.sh     # likewise LSTM, TABM, TABR
```

Inference loads checkpoint from `outputs/bench_d9r9_<model>/models/`. To produce it, uncomment the two blocks at the top of the script: training (`run_custom_train.sh`) and checkpoint prep (`prepare_custom_checkpoint4inference.sh`). Avoid using cpu to run custom predecoders, Training needs CUDA. 

Checkpoints used to be written to `QEC_model_zoo/outputs/<experiment>/models/`. Move existing ones into `outputs/<experiment>/models/` here.

#### how to run every pre-decoder on CPU?

The official Fast CNN and all four custom architectures have dedicated CPU launchers:

```bash
cd "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder"

bash QEC-PD-SURFACE-Fast-cpu.sh
```

The four custom launchers load checkpoint 10 from
`outputs/bench_d9r9_<model>/models/` by default. Override the source experiment or
checkpoint index with `EXPERIMENT_NAME` and `USE_CHECKPOINT`. They perform inference
only; custom-model training still requires CUDA.

#### how to train the official CNN? (GPU server)

```bash
cd "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder"
bash QEC-PD-D9R9-CNN.sh
```

This calls `utils/predecoder/scripts/nvidia/local_run.sh`, which requires `nvidia-smi` and CUDA. Checkpoints go to `outputs/official_train_d9r9_single_gpu_2080ti/` and logs to `outputs/logs/`.


#### result records
Results are written to `outputs/bench_**_**`; change the folder with `EXPERIMENT_NAME`.

Given the outputs, the benchmark results can be collected via: 

```bash 
python QEC_model_zoo/benchmark/utils/predecoder/analysis/extract_surface_results.py
```
The results are saved to `benchmark/surface code/NVIDIA-predecoder/outputs/bench_surface_fast_cpu_results.csv`. 
