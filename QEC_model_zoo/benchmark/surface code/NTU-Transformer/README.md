### NTU benchmarks for Surface code

Original source: https://github.com/GrahamYan/ntu-decoder/tree/main

This folder contains:

- `NTU-baselines/`: PyMatching baseline (correlated and standard matching), in `baseline.py` / `baseline.sh`
- `NTU-Transformer/`: NTU Transformer decoder, with `inference.py` / `inference.sh` for evaluation and `transformer.py` / `train.sh` for training

On macOS, the PyMatching baseline and Transformer inference run on CPU. Training needs CUDA GPUs (it uses NCCL and `torch.cuda`), so run it on the GPU server.

All commands below start from the repository root. The folder name `surface code` contains a space, so always quote the path.

#### how to install conda environment for running experiments? (macOS, Linux)

```bash
cd "QEC_model_zoo/benchmark/surface code"
conda env create -f environment-macos.yml  (or conda env create -f environment.yml)
conda activate NTU-repro
```

The scripts print `[WARN] CONDA_SH_PATH not set or not found. Skipping conda activation.` This is expected: they use whichever environment is already active.

#### how to run the PyMatching baseline?

```bash
cd "QEC_model_zoo/benchmark/surface code/NTU-baselines"

# Correlated matching
bash baseline.sh --d "3 5 7" --p 0.003 --shots 200000 --mode correlated

# Standard matching
bash baseline.sh --d "3 5 7" --p 0.003 --shots 200000 --mode standard
```

- `--d`: one or more distances. Quote the list (`--d "3 5 7"`); `--d 3 5 7` fails with `Unknown argument: 5`.
- `--p`: physical error rate (required).
- `--shots`: shots per distance (default 5,000,000).
- `--mode`: `correlated` (default) or `standard`.

Results are printed to the terminal; no file is written. The correlated command above takes about 3 s on an Apple M4.

#### how to run NTU-Transformer inference?

```bash
cd "QEC_model_zoo/benchmark/surface code/NTU-Transformer"

# Checkpoint from Hugging Face (downloads surface/d7.pth, cached in ~/.cache/huggingface)
bash inference.sh --d 7 --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
    --shots 2000 --eval_p 0.003

# With a local checkpoint
bash inference.sh --d 7 --ckpt /path/to/d7.pth --shots 2000 --eval_p 0.003
```

- Without a CUDA GPU it runs on CPU automatically (Apple MPS is not used).
- `--hf_repo` loads `surface/d{d}.pth`, so the repository must have a checkpoint for that distance.
- `--eval_p` defaults to 0.003 and `--batch_size` to 256.
- Results are appended to `eval_d{d}_results.csv` in the current directory.
- CPU is slow: 512 shots at d=7 took about 20 s on an Apple M4, including model loading. Keep `--shots` small for local checks and run large evaluations on the GPU server.

#### training (GPU server only)

`train.sh` launches `torchrun` with `--nproc_per_node=8` and the NCCL backend, so it does not run on macOS. On the server, set `CONDA_SH_PATH` to your `conda.sh` so the script can activate its environment (`tennis`), or activate the environment yourself. Edit `--nproc_per_node` in `train.sh` if the server has fewer than 8 GPUs.

```bash
cd "QEC_model_zoo/benchmark/surface code/NTU-Transformer"

# Transfer from the d=7 checkpoint on Hugging Face to d=11
bash train.sh --mode transfer \
    --hf_ckpt Dreamworldsmile/ntu-surface-code-decoder/surface/d7.pth --d 11 \
    --train_p 0.005 --eval_p 0.005 --target_high 0.98 --target_low 0.98 \
    --batch_size 32 --lr 3e-5 --max_steps 80000 --output_dir ./experiments
```

See the header of `train.sh` for training from scratch (`--mode scratch`) and resuming from a local checkpoint (`--ckpt`).
