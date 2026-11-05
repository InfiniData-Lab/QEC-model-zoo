# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra, sys, torch, os, json, numpy as np
from omegaconf import DictConfig, OmegaConf
from training.train import main as train_main
from model.factory import ModelFactory
from data.factory import DatapipeFactory
from hydra.utils import to_absolute_path
from workflows.config_validator import (
    apply_public_defaults_and_model,
    validate_public_config,
)

from training.distributed import DistributedManager

from torch.utils.data import DataLoader


def _ensure_inference_io_channels(cfg):
    # 1) Ensure out_channels matches the model’s heads (4: z_data, x_data, syn_x, syn_z)
    if not getattr(cfg.model, "out_channels", None) or cfg.model.out_channels == 0:
        cfg.model.out_channels = 4

    # 2) Infer input_channels from a single inference sample if not set
    if not getattr(cfg.model, "input_channels", None) or cfg.model.input_channels == 0:
        ds = DatapipeFactory.create_datapipe_inference(cfg)
        tmp = DataLoader(ds, batch_size=1)
        sample = next(iter(tmp))
        cfg.model.input_channels = int(sample["trainX"].shape[1])

    # 3) Keep num_filters consistent with out_channels
    if hasattr(cfg.model, "num_filters"):
        filters = list(cfg.model.num_filters)
        if filters and filters[-1] != cfg.model.out_channels:
            print(
                f"[run] Adjusting model.num_filters[-1] {filters[-1]} -> {cfg.model.out_channels}"
            )
            filters[-1] = cfg.model.out_channels
            cfg.model.num_filters = filters



def _apply_custom_model_runtime_overrides(cfg):
    """Runtime overrides for custom_model_id experiments.

    This function runs AFTER public config validation and public defaults expansion.
    Therefore it does not add unsupported fields such as train/batch_schedule to
    config_public.yaml before validation.

    It keeps the official run.py untouched and preserves the original workflow
    logging style as much as possible.
    """
    import os

    custom_model_id = getattr(cfg, "custom_model_id", None)
    # Apply runtime overrides for custom models AND for the color cascade "Model B"
    # (which has no custom_model_id but still needs the PREDECODER_TEST_* overrides).
    is_color = str(getattr(cfg, "code", "")) == "color"
    if custom_model_id is None and not is_color:
        return cfg

    def _int_env(name, default=None):
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        return int(value)

    def _float_env(name, default=None):
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        return float(value)
    

    train_epochs = _int_env("PREDECODER_TRAIN_EPOCHS")
    train_samples = _int_env("PREDECODER_TRAIN_SAMPLES")
    accumulate_steps = _int_env("PREDECODER_ACCUMULATE_STEPS", 1)

    if hasattr(cfg, "train"):
        if train_epochs is not None:
            cfg.train.epochs = train_epochs
        if train_samples is not None:
            cfg.train.num_samples = train_samples
        cfg.train.accumulate_steps = accumulate_steps

    batch_initial = _int_env("PREDECODER_BATCH_INITIAL", 16)
    batch_final = _int_env("PREDECODER_BATCH_FINAL", batch_initial)

    if hasattr(cfg, "batch_schedule"):
        cfg.batch_schedule.enabled = True
        cfg.batch_schedule.initial = batch_initial
        cfg.batch_schedule.final = batch_final
        cfg.batch_schedule.start_epoch = 0
        cfg.batch_schedule.end_epoch = 0

    val_samples = _int_env("PREDECODER_VAL_SAMPLES", 1024)
    if hasattr(cfg, "val"):
        cfg.val.num_samples = val_samples

    lr = _float_env("PREDECODER_LR")
    if lr is not None and hasattr(cfg, "optimizer"):
        cfg.optimizer.lr = lr

    optimizer_type = os.environ.get("PREDECODER_OPTIMIZER_TYPE")
    if optimizer_type:
        cfg.optimizer_type = optimizer_type

    lr = _float_env("PREDECODER_LR")
    if lr is not None and hasattr(cfg, "optimizer"):
        cfg.optimizer.lr = lr

    validation_ler = os.environ.get("PREDECODER_VALIDATION_LER", "false").lower()
    if validation_ler in {"0", "false", "no"}:
        cfg.validation_ler = False

    torch_compile = os.environ.get("PREDECODER_TORCH_COMPILE", "false").lower()
    if torch_compile in {"0", "false", "no"}:
        cfg.torch_compile = False

    print("[CustomModels] Runtime overrides applied")
    print("[CustomModels] custom_model_id =", getattr(cfg, "custom_model_id", None))
    print("[CustomModels] model_id =", getattr(cfg, "model_id", None), "code =", getattr(cfg, "code", None))

    if hasattr(cfg, "train"):
        print("[CustomModels] train.epochs =", cfg.train.epochs)
        print("[CustomModels] train.num_samples =", cfg.train.num_samples)
        print("[CustomModels] train.accumulate_steps =", cfg.train.accumulate_steps)

    if hasattr(cfg, "batch_schedule"):
        print("[CustomModels] batch_schedule.initial =", cfg.batch_schedule.initial)
        print("[CustomModels] batch_schedule.final =", cfg.batch_schedule.final)

    if hasattr(cfg, "val"):
        print("[CustomModels] val.num_samples =", cfg.val.num_samples)

    if hasattr(cfg, "optimizer"):
        print("[CustomModels] optimizer.lr =", cfg.optimizer.lr)


    # Inference / LER runtime overrides
    test_distance = _int_env("PREDECODER_TEST_DISTANCE")
    test_n_rounds = _int_env("PREDECODER_TEST_N_ROUNDS")
    test_samples = _int_env("PREDECODER_TEST_SAMPLES")
    test_batch_size = _int_env("PREDECODER_TEST_BATCH_SIZE")
    use_checkpoint = _int_env("PREDECODER_USE_CHECKPOINT")
    test_p_error = _float_env("PREDECODER_TEST_P_ERROR")
    test_meas_basis = os.environ.get("PREDECODER_TEST_MEAS_BASIS")

    if hasattr(cfg, "test"):
        if test_distance is not None:
            cfg.test.distance = test_distance
        if test_n_rounds is not None:
            cfg.test.n_rounds = test_n_rounds
        if test_samples is not None:
            cfg.test.num_samples = test_samples
        if test_batch_size is not None and hasattr(cfg.test, "dataloader"):
            cfg.test.dataloader.batch_size = test_batch_size
        if use_checkpoint is not None:
            cfg.test.use_model_checkpoint = use_checkpoint
        if test_p_error is not None:
            cfg.test.p_error = test_p_error
        if test_meas_basis:
            cfg.test.meas_basis_test = str(test_meas_basis)

        print("[CustomModels] test.distance =", cfg.test.distance)
        print("[CustomModels] test.n_rounds =", cfg.test.n_rounds)
        print("[CustomModels] test.num_samples =", cfg.test.num_samples)
        if hasattr(cfg.test, "dataloader"):
            print("[CustomModels] test.dataloader.batch_size =", cfg.test.dataloader.batch_size)
        print("[CustomModels] test.use_model_checkpoint =", getattr(cfg.test, "use_model_checkpoint", None))


    print("[CustomModels] validation_ler =", getattr(cfg, "validation_ler", None))
    print("[CustomModels] torch_compile =", getattr(cfg, "torch_compile", None))

    return cfg


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> None:
    # Early-access public release: validate public surface, then merge in hidden defaults.
    # NOTE: Validation is done BEFORE merging defaults so we can fail fast on injected fields.
    model_spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, model_spec)
    cfg = _apply_custom_model_runtime_overrides(cfg)

    torch.backends.cuda.matmul.allow_tf32 = cfg.enable_matmul_tf32
    torch.backends.cudnn.allow_tf32 = cfg.enable_cudnn_tf32

    # Dispatch by code family. surface (and its historical "surface_partition" alias)
    # keep their existing path; color uses the stim/Chromobius path.
    from qec.code_family import get_code_family

    family = get_code_family(cfg.code)
    if family.name == "surface":
        run_surface(cfg)
    elif family.name == "color":
        run_color(cfg)
    else:
        raise ValueError(f"No run dispatcher for code family {family.name!r}")


def run_surface(cfg: DictConfig):
    if cfg.workflow.task == "train":
        train_main(cfg)
    elif cfg.workflow.task == "threshold":
        raise ValueError(
            "workflow.task='threshold' has been renamed to workflow.task='inference'. "
            "Please update your config/env var to WORKFLOW=inference."
        )
    elif cfg.workflow.task == "inference":
        from evaluation.inference import run_inference
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        run_inference(model, dist.device, dist, cfg)
    elif cfg.workflow.task == "data":
        DistributedManager.initialize()
        dist = DistributedManager()
        train_loader, _ = DatapipeFactory.create_dataloader(cfg, dist.world_size, dist.rank)
        for j, dl in enumerate(train_loader):
            print(f"Batch {j}: syndrome_shape: {dl['syndrome'].shape}")
    elif cfg.workflow.task == "decoder_ablation":
        from evaluation.failure_analysis import decoder_ablation_study
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        decoder_ablation_study(model, dist.device, dist, cfg)
    elif cfg.workflow.task in ("sampling", "visualize"):
        raise ValueError(
            f"workflow.task={cfg.workflow.task!r} is not supported in the early-access public release. "
            "Supported workflows: train, inference, decoder_ablation."
        )


def run_color(cfg: DictConfig):
    """Run dispatcher for the color-code family (stim + Chromobius path).

    Mirrors run_surface's task switch. The code-family differences (color datapipe,
    Chromobius global decoder) are resolved inside train_main / run_inference via
    cfg.code, so the task dispatch itself is identical.
    """
    if cfg.workflow.task == "train":
        train_main(cfg)
    elif cfg.workflow.task == "inference":
        from evaluation.logical_error_rate_color import count_logical_errors_color
        DistributedManager.initialize()
        dist = DistributedManager()
        model = _load_model(cfg, dist)
        result = count_logical_errors_color(
            model, dist.device, dist, cfg, include_diagnostics=True, log_summary=True
        )
        if dist.rank == 0 and isinstance(result, dict):
            print("\n" + "=" * 60)
            print("Color Code Inference Results (faithful detector_input + Chromobius)")
            print("=" * 60)
            for basis, data in result.items():
                print(f"\n{basis}-basis:")
                for key, value in data.items():
                    if not isinstance(value, dict):
                        print(f"  {key}: {value}")
    else:
        raise ValueError(
            f"workflow.task={cfg.workflow.task!r} is not supported for code=color. "
            "Supported color-code workflows: train, inference."
        )


def find_best_model(path, *, rank: int = 0):
    if rank == 0:
        print(f"Searching for best model in: {path}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    max_value = -1  # Start with -1 to include epoch 0
    best_file = None
    model_files = []
    # Named .pt files without epoch numbers (e.g. Ising-Decoder-SurfaceCode-1-Fast.pt)
    named_pt_files = []

    for filename in os.listdir(path):
        if not filename.endswith(".pt"):
            continue
        if filename.startswith("PreDecoderModelMemory_"):
            try:
                value = float(filename.split(".")[2])  # Gets epoch number
                model_files.append((filename, value))
                if value > max_value:
                    max_value = value
                    best_file = filename
            except (IndexError, ValueError) as e:
                print(f"Warning: could not parse epoch from filename {filename}: {e}")
        else:
            named_pt_files.append(filename)

    # Fall back to named .pt files when no epoch-numbered checkpoints are present
    if best_file is None and named_pt_files:
        named_pt_files.sort()
        best_file = named_pt_files[-1]
        model_files = [(f, None) for f in named_pt_files]

    if rank == 0:
        print(f"Found {len(model_files)} model file(s):")
        for filename, epoch in sorted(model_files, key=lambda x: (x[1] is None, x[1] or 0)):
            marker = "*" if filename == best_file else " "
            epoch_str = str(epoch) if epoch is not None else "n/a"
            print(f"  [{marker}] {filename} (epoch {epoch_str})")

    if best_file is None:
        raise FileNotFoundError(
            f"No valid model checkpoint files found in {path}\n"
            f"Expected .pt files (e.g. Ising-Decoder-SurfaceCode-1-Fast.pt or "
            f"PreDecoderModelMemory_*.pt).\n"
            f"Hint: download the pretrained weights and place them in this directory, "
            f"or set model_checkpoint_file in your config to an explicit path."
        )

    best_model_path = os.path.join(path, best_file)
    if rank == 0:
        epoch_str = str(max_value) if max_value >= 0 else "n/a"
        print(f"Selected best model: {best_file} (epoch {epoch_str})")

    return best_model_path


def _resolve_dir(path: str) -> str:
    """Return an absolute version of path, resolving relative paths from the repo root."""
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def _load_state_dict_from_pt(model_path: str, device) -> dict:
    """Load a state dict from a .pt checkpoint, handling multiple saved formats.

    Supports:
    - bare state dict (keys are layer names)
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    Also strips the DDP "module." prefix if present.
    """
    raw = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: expected a dict, got {type(raw).__name__}")
    return {
        (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }


def _load_model(cfg, dist):
    if dist.rank == 0:
        print(f"Loading model for task: {cfg.workflow.task}")

    _ensure_inference_io_channels(cfg)

    # SafeTensors path: load fp16/fp32 model from SafeTensors file
    safetensors_path = os.environ.get("PREDECODER_SAFETENSORS_CHECKPOINT", "").strip()
    if safetensors_path:
        from utils.export.safetensors_utils import load_safetensors
        if dist.rank == 0:
            print(f"Loading model from SafeTensors: {safetensors_path}")

        # Auto-detect model_id from SafeTensors metadata (don't override with config)
        model, metadata = load_safetensors(
            safetensors_path,
            model_id=None,
            device=str(dist.device),
        )
        if dist.rank == 0:
            loaded_model_id = metadata.get("model_id", "unknown")
            dtype = metadata.get("quant_format", "fp32")
            receptive_field = metadata.get("receptive_field", "unknown")
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  model_id: {loaded_model_id} (from SafeTensors metadata)")
            print(f"  receptive_field: {receptive_field}")
            print(f"  dtype: {dtype}")
            print(f"  parameters: {param_count:,}")

            # Warn if config model_id doesn't match file metadata
            config_model_id = getattr(cfg, "model_id", None)
            if config_model_id is not None and str(config_model_id) != str(loaded_model_id):
                print(
                    f"  Warning: config model_id={config_model_id} differs from "
                    f"file model_id={loaded_model_id}; using {loaded_model_id}"
                )

        if metadata.get("quant_format") == "fp16":
            cfg.enable_fp16 = True
        return model

    # Direct file path override (for named pretrained models without epoch numbers)
    model_checkpoint_file = getattr(cfg, 'model_checkpoint_file', None)
    if model_checkpoint_file:
        model_checkpoint_file = _resolve_dir(str(model_checkpoint_file))
        if not os.path.exists(model_checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint_file}")
        if dist.rank == 0:
            print(f"Loading model from: {model_checkpoint_file}")
        model = ModelFactory.create_model(cfg).to(dist.device)
        if cfg.enable_fp16:
            model = model.half()
        state_dict = _load_state_dict_from_pt(model_checkpoint_file, dist.device)
        model.load_state_dict(state_dict)
        if dist.rank == 0:
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model loaded ({param_count:,} parameters)")
        return model

    model = ModelFactory.create_model(cfg).to(dist.device)

    if cfg.enable_fp16:
        model = model.half()
        if dist.rank == 0:
            print("Model converted to float16 for fp16 inference")

    # Determine model directory
    # Priority: 1) model_checkpoint_dir (for inference configs)
    #           2) cfg.output/models (for training configs)
    model_checkpoint_dir = getattr(cfg, 'model_checkpoint_dir', None)
    use_checkpoint = getattr(cfg.test, 'use_model_checkpoint', -1)

    if use_checkpoint == -1:
        model_dir = _resolve_dir(
            os.path.join(model_checkpoint_dir, "best_model")
            if model_checkpoint_dir else f"{cfg.output}/models/best_model"
        )
        if dist.rank == 0:
            print(f"Loading best model from: {model_dir}")

        # Fallback: older runs may not have a best_model/ folder
        if not os.path.isdir(model_dir):
            fallback_dir = _resolve_dir(
                model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
            )
            if dist.rank == 0:
                print(f"best_model/ not found; falling back to: {fallback_dir}")
            model_dir = fallback_dir

        model_path = find_best_model(model_dir, rank=dist.rank)
    else:
        checkpoint_dir = _resolve_dir(
            model_checkpoint_dir if model_checkpoint_dir else f"{cfg.output}/models"
        )
        if dist.rank == 0:
            print(f"Loading checkpoint {use_checkpoint} from: {checkpoint_dir}")

        # Prefer any PreDecoderModelMemory_* file ending with .0.{use_checkpoint}.pt
        target_suffix = f".0.{use_checkpoint}.pt"
        checkpoint_filename = None
        try:
            for f in os.listdir(checkpoint_dir):
                if f.startswith("PreDecoderModelMemory_") and f.endswith(target_suffix):
                    checkpoint_filename = f
                    break
        except OSError:
            pass
        if checkpoint_filename is None:
            checkpoint_filename = f"PreDecoderModelMemory_v1.0.{use_checkpoint}.pt"
        model_path = os.path.join(checkpoint_dir, checkpoint_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if dist.rank == 0:
        print(f"Loading model parameters from: {model_path}")

    state_dict = _load_state_dict_from_pt(model_path, dist.device)
    model.load_state_dict(state_dict)

    if dist.rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model loaded ({param_count:,} parameters)")

    return model


if __name__ == "__main__":
    run()
