# QEC Model Zoo
This repository contains the code and experiments accompanying the paper **"QEC Model Zoo: Democratizing AI-enhanced Quantum Error Correction,"** accepted for publication at NeurIPS 2026.

## Repository Structure
This section of the documentation explains the most important files and directories of the repository.

### Experiments
The [`experiments`](experiments/) directory contains configurations for full-decoders (also referred to as models).
Training experiments train the decoders and find (near) optimal hyperparameters through hyperparameter tuning using the Optuna library [1].
After training, model checkpoints and parameters are saved in the [`experiments/model_checkpoints`](experiments/model_checkpoints/) directory. 
These checkpoints and parameters are then used to configure the inference experiments.
After experiments are run, results will be saved to the [`experiments/results`](experiments/results/) directory.

### Datasets & Croissant Metadata
Pre-processed dataset files reside in the [`datasets`](datasets/) directory.
If this directory does not exist in the root of the project or is incomplete, you can download it through [Kaggle](https://www.kaggle.com/datasets/infinidata/qec-model-zoo-data) make sure the [`datasets`](datasets/) folder is located in the root of the project and has the following structure:
``` 
datasets/
	croissant/
	dicarlo_rep_d3/
	google_color/
	...
	dataset_catalog.yaml
	README.md
```

Alternatively, you can download the pre-processed datasets from Kaggle through the CLI:
```bash
pip install kaggle
kaggle datasets download infinidata/qec-model-zoo-data -p datasets --unzip
```

For each dataset used to reproduce our results, we record its metadata, including licensing information in a croissant metadata file.
This unified representation facilitates data processing for downstream full-decoders.
Croissant metadata files,  including references to the dataset source (citation) can be found in the [`datasets/croissant`](datasets/croissant/) directory.
The [`dataset_catalog.yaml`](datasets/dataset_catalog.yaml) file is used to generate each croissant file using the [`QEC_model_zoo/scripts/generate_croissant.py`](QEC_model_zoo/scripts/generate_croissant.py) script.
The [`README.md`](datasets/README.md) file provides additional details about datasets.

### Source Code
The source code for all models and experiments is located in the [`QEC_model_zoo`](QEC_model_zoo/) directory. The source code is modularized, with each module having separate responsibilities (e.g. data for dataset loading and processing).
[`main.py`](QEC_model_zoo/main.py) is the most important file for reproducing results as it is the entrypoint of the application.

### Benchmark Scripts
To reproduce the results in our work, we provide shell scripts, organized per QEC code family in the [`QEC_model_zoo/benchmark`](QEC_model_zoo/benchmark/) directory.
The QEC model zoo supports the following code families:
- BB code
- Color code
- Repetition code
- Surface code

### DAIC Scripts
Results for full-decoders are obtained using the Delft AI Cluster (DAIC) [2]. The [`daic_scripts`](daic_scripts/) directory contains SLURM scripts to run experiments on DAIC.

## Full-Decoders
### Supported Models & Model Cards
For each of the full-decoders included in the model zoo, we record a model card, including a citation to the source, and the license of the implementation if applicable, model checkpoints generated from the model zoo may be licensed separately.
For each model card, see:
- [CatBoost](QEC_model_zoo/model/full_decoder/model_cards/catboost.yaml)
- [LightGBM](QEC_model_zoo/model/full_decoder/model_cards/lightgbm.yaml)
- [XGBoost](QEC_model_zoo/model/full_decoder/model_cards/xgboost.yaml)
- [MLP](QEC_model_zoo/model/full_decoder/model_cards/mlp.yaml)
- [TabM](QEC_model_zoo/model/full_decoder/model_cards/tabm.yaml)
- [TabR](QEC_model_zoo/model/full_decoder/model_cards/tabr.yaml)
- [TabPFNv2](QEC_model_zoo/model/full_decoder/model_cards/tabpfnv2.yaml)
- [TabICLv2](QEC_model_zoo/model/full_decoder/model_cards/tabiclv2.yaml)
- [LSTM](QEC_model_zoo/model/full_decoder/model_cards/lstm.yaml)

### Reproducing Full Decoder-Results
To reproduce full-decoder results for repetition code, surface code, and color code, we run the following benchmark scripts, make sure training scripts are run before the inference scripts, or that you have model checkpoints saved already.

#### Setting up Environment
It is recommended to use Python 3.10 and a virtual environment (or conda) with the following requirements installed

```bash
pip install -r requirements.txt
```

Furthermore, a GPU with CUDA enabled is required to reproduce full-decoder results. Make sure that the pytorch version and index url in the [`requirements.txt`](requirements.txt) file is aligned with your CUDA version, our requirements file is set up for CUDA 12.4 and Python 3.10.
Note that for the usage of the full-decoder TabPFNv2, a token is required and should be set in your execution environment as an environment variable, for instructions on how to obtain a token, see [https://platform.priorlabs.ai/playground](https://platform.priorlabs.ai/playground).
Also make sure that the pre-processed datasets are downloaded as described in section "Repository Structure" of this file.

#### Running Training Experiments
To reproduce our results, you can either use the pre-trained model checkpoints available in the [`experiments/model_checkpoints/`](experiments/model_checkpoints/) directory or train the models from scratch.

The model checkpoints are stored using [Git Large File Storage (Git LFS)](https://git-lfs.com/). To ensure the checkpoints are downloaded correctly, install Git LFS and run:

```bash
git lfs install
git lfs pull
```

The remainder of this section describes the procedure for training the models from scratch.

To train full-decoders on all datasets, you can run the following shell scripts:
- Repetition code: 
	```bash
	./"QEC_model_zoo/benchmark/repetition_code/full_decoder/train_full_decoders_repetition_code.sh"
	```
- Color code: 
	```bash
	./"QEC_model_zoo/benchmark/color code/full_decoder/train_full_decoders_color_code.sh"
	```
- Surface code: 
   ```bash
   ./"QEC_model_zoo/benchmark/surface code/full_decoder/train_full_decoders_surface_code.sh"
   ```

Model checkpoints are saved to the [`experiments/model_checkpoints`](experiments/model_checkpoints/) directory.

#### Running Inference Experiments
To run inference experiments for full-decoders on all datasets, you can run the following scripts:
- Repetition code: 
	```bash
	./"QEC_model_zoo/benchmark/repetition_code/full_decoder/inference_full_decoders_repetition_code.sh"
	```
- Color code: 
	```bash
	./"QEC_model_zoo/benchmark/color code/full_decoder/inference_full_decoders_color_code.sh"
	```
- Surface code: 
   ```bash
   ./"QEC_model_zoo/benchmark/surface code/full_decoder/inference_full_decoders_surface_code.sh"
   ```

Results are saved to the [`experiments/results`](experiments/results/) directory.

#### Running Feature Ablation
For the DiCarlo repetition code dataset, we perform a feature group ablation study; results for this can be reproduced by running the following script:
``` bash
./"QEC_model_zoo/benchmark/repetition_code/full_decoder/feature_ablation_full_decoders_repetition_code.sh"
```
Results of the feature ablation are saved in the [`experiments/results/ablations`](experiments/results/ablations/) directory.

### Result Analysis
Analysis of inference experiment results for full-decoders is done in the [`full_decoder_results_analysis.ipynb`](full_decoder_results_analysis.ipynb) file.
This reports both evaluation metrics and run-time measurement statistics in the [`experiments/results`](experiments/results/) directory.
Additional analysis including pareto-frontier plotting can be found in the [`QEC_model_zoo/analysis/QEC_full_decoders_Pareto_frontiers.ipynb`](QEC_model_zoo/analysis/QEC_full_decoders_Pareto_frontiers.ipynb) file.
Figures are saved to the [`QEC_model_zoo/analysis/figures`](QEC_model_zoo/analysis/figures/) directory.

### Hardware & Software specifications
All full-decoder experiments are run in a virtual environment with Python version 3.10 and CUDA 12.4, note that running the experiments requires CUDA.
The hardware we use for full-decoder experiments is a single NVIDIA GPU of the Delft AI Cluster [2] with the following specifications:
- Model: NVIDIA L40
- Architecture: Ada Lovelace
- CUDA cores: 18176
- Memory: 49152 MiB

## Pre-Decoders
The experimental code for the predecoders is sourced from https://github.com/NVIDIA/Ising-Decoding. 

Benchmark experiments for color code can be found at `QEC_model_zoo/benchmark/color code/NVIDIA-predecoder`, while those for surface code are located at `QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder`, the relevant functional implementations can be found at `QEC_model_zoo/benchmark/utils/predecoder`. Each directory contains a `README` file detailing the specific experimental setup. In our case, we first installed miniconda and then created an environment named `Ising-Decoding` for experiments. 

### Tested Models & Model Cards
The following models are tested in our benchmarks, which are publicly accessible through Hugging Face. 
- Surface code: https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast
- Color code: https://huggingface.co/nvidia/Ising-Decoder-ColorCode-1-Fast

### Run experiments
Note that these models are accessed-controlled with license. To conduct the inference experiments, it requires downloading these models. 
When the conda environment and pre-trained pre-decoders are ready, you can run the following scripts: 

Surface code: 
```bash
cd "QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder"
bash QEC-PD-SURFACE-Fast-cpu.sh
```

Color code:
```bash
cd "QEC_model_zoo/benchmark/color code/NVIDIA-predecoder"
bash QEC-PD-COLOR-ModelB-cpu.sh
```
More details can be found at `QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder/README.md` and `QEC_model_zoo/benchmark/color code/NVIDIA-predecoder/README.md`, respectively. 

### Result Analysis 
The `outputs` folders contain logs from runs conducted across different experimental platforms in the virtual environment called `Ising-Decoding`. Specifically, the experimental results are stored within these `outputs` directories collected from a workstation equipped with an NVIDIA GeForce RTX 2080 Ti GPU (11 GB GDDR6, driver 525.147.05), an Intel Core i9-10900KF CPU (3.70 GHz, 10 cores/20 threads), and 31 GiB of system RAM (running Python 3.11.15 and CUDA 12.0), or an Apple M4 chip (10-core CPU, 10-core GPU, 16 GB of unified memory, and 500 GB SSD storage). The latest results of the predecoder inference experiments are saved in `QEC_model_zoo/benchmark/surface code/NVIDIA-predecoder/outputs/bench_surface_fast_cpu_results.csv` and `QEC_model_zoo/benchmark/color code/NVIDIA-predecoder/outputs/bench_color_modelB_cpu_d5_results.csv`, which are collected from the workstation on the CPU backend. 
Additional analysis code for pre-decoders can be found in the [`QEC_model_zoo/analysis/QEC_predecoders_statistics.ipynb`](QEC_model_zoo/analysis/QEC_predecoders_statistics.ipynb) file.

## NTU decoders
The experimental code for the NTU decoders is sourced from https://github.com/GrahamYan/ntu-decoder.

 Benchmark experiments for BB code can be found at `QEC_model_zoo/benchmark/BB code/NTU-Transformer` and `QEC_model_zoo/benchmark/BB code/NTU-Neural-BP`, while those for surface code are located at `QEC_model_zoo/benchmark/surface code/NTU-Transformer`. 
 
 ### Tested Models & Model Cards
The following models are tested in our benchmarks from https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder 

BB72 Transformer checkpoint: 

- Dreamworldsmile/ntu-surface-code-decoder/bb/bb72_transformer.pt 
- Dreamworldsmile/ntu-surface-code-decoder/bb/neural_bp_bb72.pt 

Surface code Transformer checkpoint: 
- Dreamworldsmile/ntu-surface-code-decoder/surface/d7.pth

### Run inference experiments
These models can be downloaded with the MIT license. In `QEC_model_zoo/benchmark/BB code/README.md`, we include the instructions about running inference experiments for BB code. The commands download the checkpoints from Hugging Face (`bb/bb72_transformer.pt` and `bb/neural_bp_bb72.pt`), which is then cached in `~/.cache/huggingface` (see the saved run logs).

### Result Analysis
 The BB code results of the NTU decoder experiments are saved in `QEC_model_zoo/benchmark/BB code/NTU-Transformer/bb72_transformer_eval_r6_p0.005.csv` and `QEC_model_zoo/benchmark/BB code/NTU-Neural-BP/bb72_neuralbp_eval_r6_p0.005.csv`, respectively. The `QEC_model_zoo/benchmark/surface code/NTU-Transformer/eval_d7_results.csv` stores the surface-code experimental results. These results are collected from a workstation equipped with an NVIDIA GeForce RTX 2080 Ti GPU (11 GB GDDR6, driver 525.147.05), an Intel Core i9-10900KF CPU (3.70 GHz, 10 cores/20 threads), and 31 GiB of system RAM (running Python 3.11.15 and CUDA 12.0). 

We note that the quantity reported as `LER` in the saved result files is in fact (1 - Block Acc), i.e., the logical error probability accumulated over all R=6 rounds. To obtain the per-round logical error rate $$\epsilon_L$$, we convert the LER per round by the formula $$\epsilon_L = \frac{1}{2}[1-(1-2P_{\mathrm{block}}(R))^{\frac{1}{R}}]$$. Additional analysis code for NTU decoders can be found in the [`QEC_model_zoo/analysis/QEC_ntu_decoders_LER per round.ipynb`](QEC_model_zoo/analysis/QEC_ntu_decoders_LER per round.ipynb) file.

## GNN 
The experimental code for reproducing GNN decoding results included in our work can be found in the [`QEC_model_zoo/gnn_qec-main`](QEC_model_zoo/gnn_qec-main/) directory.
To obtain these results we:
1. In a terminal navigate to the [`QEC_model_zoo/gnn_qec-main`](QEC_model_zoo/gnn_qec-main/) directory.
2. Run the following shell script:
	```shell
	./scripts/run_model_zoo_experiment.sh
	```
 3. This outputs results to the [`QEC_model_zoo/gnn_qec-main/outputs/metrics`](QEC_model_zoo/gnn_qec-main/outputs/metrics/) directory.

Additional code for analyzing and formatting these results can be found in the [`QEC_model_zoo/analysis/QEC_gnn_decoders_analysis.ipynb`](QEC_model_zoo/analysis/QEC_gnn_decoders_analysis.ipynb) file.

## Licensing

Software and model checkpoints introduced through this code repository are licensed under the Apache License, Version 2.0, subject to the Commons Clause License Condition v1.0 (see the [`LICENSE`](LICENSE) file in this repository).
Externally sourced datasets, software, and model checkpoints remain subject to their respective licenses.

## Citation

If you use QEC Model Zoo, its datasets, or its pre-trained models in your research, please cite our paper:

**Wang, Yuandou; de Groot, Max; Patwardhan, Aadi; Geerts, Floris; and Hai, Rihan.** (2026). **QEC Model Zoo: Democratizing AI-enhanced Quantum Error Correction.** *The Fortieth Annual Conference on Neural Information Processing Systems (NeurIPS 2026), Evaluations and Datasets Track.* Accepted, to appear.

**BibTeX:**

```bibtex
@inproceedings{wang2026qecmodelzoo,
  title={{QEC} Model Zoo: Democratizing {AI}-enhanced Quantum Error Correction},
  author={Wang, Yuandou and de Groot, Max and Patwardhan, Aadi and Geerts, Floris and Hai, Rihan},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year={2026},
  note={Accepted at NeurIPS 2026; to appear}
}
```

When using individual datasets from this collection, please also cite their original publications, as documented in the corresponding Croissant metadata files.

## References
[1] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
	Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD. <br>
[2] Delft AI Cluster (DAIC). (2024). The Delft AI Cluster (DAIC), RRID:SCR_025091. https://doi.org/10.4233/rrid:scr_025091. <br>
[3] Christopher Chamberland, Jan Olle, Muyuan Li, Scott Thornton, and Igor Baratta. Fast and accurate AI-based pre-decoders for surface codes, 2026. URL https://arxiv.org/abs/2604.12841. <br>
[4] Jan Olle, Christopher Chamberland, Muyuan Li, and Igor Baratta. Fast and accurate AI-based pre-decoders for color codes. arXiv preprint arXiv:2607.10058, 2026. <br>
[5] Ge Yan, Shanchuan Li, Shiyi Xiao, Pengyue Ma, Hanyan Cao, Feng Pan, and Yuxuan Du. Efficient foundation decoders for fault-tolerant quantum computing. arXiv preprint arXiv:2606.27119, 2026.
