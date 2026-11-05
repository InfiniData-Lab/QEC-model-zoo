### NTU benchmarks for BB code

Original source: https://github.com/GrahamYan/ntu-decoder/tree/main

This folder contains benchmarks for the [[72,12,6]] bivariate bicycle code (BB72):

- `NTU-baselines/`: BP-OSD and Relay BP baselines, in `baseline.py`, launched by `run_baseline_bb72.sh`
- `NTU-Transformer/`: NTU Transformer decoder, in `transformer.py` (`train` / `eval`), with training scripts `train_transformer_bb72.sh` and `transfer_transformer_bb144.sh`
- `NTU-Neural-BP/`: Neural BP decoder, in `neural_bp.py` (`train` / `eval` / `generate-dems`), with training script `train_neural_bp.sh`

On macOS, the baselines and inference for both neural decoders run on CPU. Training needs CUDA GPUs (it uses NCCL and `torch.cuda`), so run it on the GPU server.

All commands below start from the repository root. The folder name `BB code` contains a space, so always quote the path.

#### how to install conda environment for running experiments? (macOS, Linux)

```bash
cd "QEC_model_zoo/benchmark/BB code"
conda env create -f environment-macos.yml (or conda env create -f environment.yml)
conda activate NTU-repro
```

`environment.yml` is the Linux/CUDA environment for the GPU server (named `tennis`). `ldpc` and `relay_bp` are commented out there; uncomment them to run the baselines on the server.

#### how to run BB code baselines?

```bash
cd "QEC_model_zoo/benchmark/BB code/NTU-baselines"

# BP-OSD
METHOD=bposd P_VALUES="0.005" NUM_SAMPLES=80 WORKERS=8 CHUNK_SAMPLES=10 \
    bash run_baseline_bb72.sh

# Relay BP
METHOD=relaybp P_VALUES="0.005" NUM_SAMPLES=1000 bash run_baseline_bb72.sh
```

Settings are passed as environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `METHOD` | `bposd` | `bposd` or `relaybp` |
| `P_VALUES` | `"0.005 0.004 0.003 0.002 0.001"` | Physical error rates (quoted, space-separated) |
| `NUM_SAMPLES` | `5000` | Samples per error rate |
| `SEED` | `20260528` | Random seed |
| `DEM_DIR` | `data/dems` | Folder with detector error models |
| `OUT` | `experiments/baselines/bposd_bb72.csv` or `relaybp_2d-reduced_bb72.csv` | Output CSV |
| `WORKERS` | `24` | BP-OSD only: worker processes |
| `CHUNK_SAMPLES` | `1000` | BP-OSD only: samples per worker task |
| `MAX_ITER`, `OSD_ORDER` | `12`, `10` | BP-OSD only: BP iterations and OSD order |
| `PRESET` | `2d-reduced` | Relay BP only: `2d-reduced` or `full` |
| `CHUNK_SIZE` | `1000` | Relay BP only: samples per batch |

Notes for macOS:

- BP-OSD sends each chunk of `CHUNK_SAMPLES` samples to one worker. With the defaults (`CHUNK_SAMPLES=1000`, `NUM_SAMPLES=5000`) at most 5 workers are busy, and with `NUM_SAMPLES` of 1000 or less only one is. Set `CHUNK_SAMPLES` to about `NUM_SAMPLES / WORKERS`. The default of 24 workers is for the server; use `WORKERS=8` on a 10-core Mac.
- CPU runs are slow. On an Apple M4, the BP-OSD command above took about 11 s (about 130 ms per sample with 8 workers, so 5000 samples take about 11 min per error rate). The Relay BP command took about 26 s (about 26 ms per sample).
- The output CSV is overwritten on every run. Set `OUT` to keep earlier results.
- Only the p=0.005 detector error model is included in `data/dems`. Missing ones are generated on the first run, or ahead of time with:

  ```bash
  python baseline.py generate-dems --ps 0.001 0.002 0.003 0.004
  ```

#### how to run inference of NTU neural decoders?

Both commands download their checkpoint from Hugging Face (`bb/bb72_transformer.pt` and `bb/neural_bp_bb72.pt`), which is then cached in `~/.cache/huggingface`. For a local checkpoint, replace `--hf_repo ...` with `--ckpt_path /path/to/file.pt`.

NTU-Transformer on [[72,12,6]]:

```bash
cd "QEC_model_zoo/benchmark/BB code/NTU-Transformer"
python transformer.py eval --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
    --shots 500 --device cpu
```

NTU-Neural-BP on [[72,12,6]]:

```bash
cd "QEC_model_zoo/benchmark/BB code/NTU-Neural-BP"
python neural_bp.py eval --block_size 72 --p 0.005 \
    --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
    --dem_path ../NTU-baselines/data/dems/72_12_6_0.005.dem \
    --shots 100 --device cpu
```

- The Transformer defaults already match BB72: `--torus_l 6 --torus_m 6 --rounds 6 --p 0.005 --d_model 512 --n_heads 8`.
- `--dem_path` reuses the detector error model from `NTU-baselines`. Without it, `neural_bp.py` generates `data/ldpc/72_12_6_<p>.dem` in the current folder.
- `--batch_size` defaults to 64 (Transformer) and 128 (Neural-BP).
- CPU inference is slow, especially for Neural-BP. On an Apple M4, 500 Transformer shots took about 80 s (about 7 samples/s), and 1000 Neural-BP shots took about 47 min (about 0.35 samples/s). Run large evaluations on the GPU server.
- Results are appended to `bb72_transformer_eval_r6_p0.005.csv` and `bb72_neuralbp_eval_r6_p0.005.csv` in the current folder.
- `inference.sh` in this folder still uses the upstream `codes/BB/...` paths and does not work here. Use the commands above instead.

#### training (GPU server only)

These scripts launch `torchrun` with the NCCL backend, so they do not run on macOS.

```bash
cd "QEC_model_zoo/benchmark/BB code/NTU-Transformer"

# BB72 from scratch
NPROC_PER_NODE=8 bash train_transformer_bb72.sh

# Transfer a BB72 checkpoint to BB144
SOURCE_CKPT=experiments/<run>/ckpts/bb72_transformer_best.pt bash transfer_transformer_bb144.sh
```

```bash
cd "QEC_model_zoo/benchmark/BB code/NTU-Neural-BP"
BLOCK_SIZE=72 P=0.005 bash train_neural_bp.sh
```

- The Transformer scripts use `NPROC_PER_NODE` GPUs (default 8). `train_neural_bp.sh` counts GPUs from `CUDA_VISIBLE_DEVICES` or `nvidia-smi`.
- Checkpoints and logs are written under `experiments/` in each folder.
- If `SOURCE_CKPT` is not set, `transfer_transformer_bb144.sh` uses the latest `bb72_transformer_best.pt` under `experiments/`.
