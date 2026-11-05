# run baselines on original 3D CNN model
export HYDRA_FULL_ERROR=1
# WORKFLOW=train EXPERIMENT_NAME=baseline_train_check bash code/scripts/local_run.sh
# WORKFLOW=inference EXPERIMENT_NAME=baseline_train_check bash code/scripts/local_run.sh

# Disabled after moving the engine to benchmark/utils/predecoder: run from QEC_model_zoo,
# these would recreate stale copies of the engine files at their old locations.
# git restore conf/config_public.yaml 2>/dev/null || true
# git restore workflows/run.py 2>/dev/null || true
# git restore model/factory.py 2>/dev/null || true
# git restore workflows/config_validator.py 2>/dev/null || true
# git restore model/registry.py 2>/dev/null || true
# git restore "training/train.py" 2>/dev/null || true


# run 
# export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export GPUS=1
export TORCH_COMPILE=0
# export WORKFLOW=inference
export DISTANCE=9
export N_ROUNDS=9
export EXPERIMENT_NAME=pretrained_2080ti_CNN

mkdir -p outputs/${EXPERIMENT_NAME}/models
# mkdir -p outputs/${EXPERIMENT_NAME}/models/best_model

# Copy original pretrained model
cp models/Ising-Decoder-SurfaceCode-1-Fast.pt \
   outputs/${EXPERIMENT_NAME}/models/Ising-Decoder-SurfaceCode-1-Fast.pt

# Also create compatible names expected by the original loader
# cp models/Ising-Decoder-SurfaceCode-1-Accurate.pt \
   # outputs/${EXPERIMENT_NAME}/models/Ising-Decoder-SurfaceCode-1-Accurate.pt

# cp models/Ising-Decoder-SurfaceCode-1-Accurate.pt \
   # outputs/${EXPERIMENT_NAME}/models/best_model/PreDecoderModelMemory_v1.0.1.pt

# Run inference
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW=inference EXPERIMENT_NAME=pretrained_2080ti_CNN bash "${SCRIPT_DIR}/../../utils/predecoder/scripts/nvidia/local_run.sh"
# # bash code/scripts/local_run.sh