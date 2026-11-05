---
license: mit
library_name: pytorch
tags:
  - qec
  - quantum-error-correction
  - bivariate-bicycle
  - quantum-ldpc
  - neural-decoder
  - transformer
  - transfer-learning
  - cpu-inference
---

# NTU-Transformer for the [[72,12,6]] BB code

This NTU-Transformer variant decodes the 12 logical observables of the
`[[72,12,6]]` bivariate-bicycle (BB) code from repeated noisy syndrome rounds.

- **Weights:** [Dreamworldsmile/ntu-surface-code-decoder](https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder)
- **Source:** [GrahamYan/ntu-decoder](https://github.com/GrahamYan/ntu-decoder)
- **Local implementation:** [`ntu-decoders/codes/BB/transformer.py`](../../../ntu-decoders/codes/BB/transformer.py)

## Checkpoint selection

| Hub filename | Recommended use |
|---|---|
| `bb/bb72_transformer.pt` | high physical error rates and transfer learning |
| `bb/bb72_transformer_lowp_finetuned.pt` | inference in the ultra-low-p regime |

The checkpoints contain `model_state` and may include training step, block
accuracy, and output-convention metadata. The original high-p checkpoint should
be used for the local `p=0.005` benchmark and as the source for BB72-to-BB144
transfer.

## Model description

`AlphaQubitV2_BB` adapts the NTU Transformer backbone to BB-code detector
geometry and multiple logical observables. The benchmark configuration uses
`d_model=512`, eight attention heads, representative logical anchors, six rounds,
and the `[[72,12,6]]` code defined by the repository's BB construction.

## Intended use

- Research evaluation of neural decoding for the supported BB code.
- Measurement of block LER and per-logical accuracy.
- Initialization of transfer learning to `[[144,12,12]]`.

The checkpoint is convention-sensitive. Code polynomials, detector ordering,
logical representatives, rounds, and output transforms must match the loader.

## CPU inference

From `QEC_model_zoo/ntu-decoders`:

```bash
conda activate NTU-repro

bash inference.sh \
  --code bb \
  --model transformer \
  --block_size 72 \
  --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
  --shots 10000 \
  --p 0.005 \
  --device cpu \
  --batch_size 16
```

## Local evaluation

Five local runs at `p=0.005`, six rounds, and 10,000 shots per run produced:

| Metric | Result |
|---|---:|
| Total shots | 50,000 |
| Pooled block LER | **0.07966 (3,983/50,000)** |
| Block accuracy | 0.92034 |
| Run-wise block LER | 0.0797 +/- 0.0032 |
| Mean per-logical accuracy | 0.9735 +/- 0.0011 |
| Worst-logical accuracy | 0.9626 +/- 0.0025 |

A block is counted as correct only when all 12 logical predictions are correct.
These are local reproduction results from
`ntu-decoders/bb72_transformer_eval_r6_p0.005.csv`, not an independent validation
of every claim in the upstream model card.

## Training data

Training examples are generated synthetically from the BB-code noise and
syndrome model. They contain no natural-language or personal data. Exact training
schedules and low-p fine-tuning ranges should be taken from the upstream paper,
source, and checkpoint provenance.

## Limitations

- Results are specific to `[[72,12,6]]`, six rounds, and the implemented output convention.
- Use of the wrong high-p/low-p checkpoint may reduce accuracy.
- Block LER cannot be compared directly with per-logical or surface-code LER.
- CPU evaluation is supported but substantially slower than CUDA evaluation.
- This research model is not a reliability guarantee for quantum hardware.

## License and citation

The upstream Hugging Face repository declares the MIT License. See the
[Surface model card](../NTU-Transformer-Surface/README.md#license-and-citation)
for the suggested citation.
