---
license: mit
library_name: pytorch
tags:
  - qec
  - quantum-error-correction
  - surface-code
  - neural-decoder
  - transformer
  - transfer-learning
  - cpu-inference
---

# NTU-Transformer for rotated surface codes

NTU-Transformer is a neural quantum-error-correction decoder from the Neural
Transfer Unification (NTU) framework. This variant maps repeated syndrome
measurements from rotated surface-code memory experiments to a logical-error
prediction.

- **Weights:** [Dreamworldsmile/ntu-surface-code-decoder](https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder)
- **Source:** [GrahamYan/ntu-decoder](https://github.com/GrahamYan/ntu-decoder)
- **Project:** [NTU Decoder](https://grahamyan.github.io/ntu-decoder/)
- **Local implementation:** [`ntu-decoders/codes/Surface`](../../../ntu-decoders/codes/Surface)

## Checkpoints

| Distance | Hub filename | Training lineage |
|---:|---|---|
| 7 | `surface/d7.pth` | trained from scratch |
| 11 | `surface/d11.pth` | transferred from d=7 |
| 15 | `surface/d15.pth` | transferred from d=11 |
| 19 | `surface/d19.pth` | transferred from d=15 |
| 23 | `surface/d23.pth` | transferred from d=19 |
| 25 | `surface/d25.pth` | transferred from d=23 |

Each checkpoint contains `model_state`, `d`, `rounds`, and `step`. Select the
checkpoint whose distance matches the generated circuit.

## Model description

The PyTorch `AlphaQubitV2` implementation uses a 512-dimensional latent space
and eight attention heads. Its QEC-specific components include scalable STEM
syndrome embeddings, 2D rotary positional embeddings, interleaved recurrent and
self-attention blocks, and a cross-attention logical readout.

## Intended use

- Research evaluation of neural decoding for rotated surface-code memory circuits.
- Reproduction of logical error rate (LER) experiments at supported distances.
- Transfer-learning research across code distances.

It is not a calibrated reliability guarantee for deployed fault-tolerant hardware.
Inputs must follow the detector ordering, circuit construction, number of rounds,
and noise conventions used by the accompanying implementation.

## CPU inference

From `QEC_model_zoo/ntu-decoders`:

```bash
conda activate NTU-repro

bash inference.sh \
  --code surface \
  --d 7 \
  --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
  --shots 100000 \
  --eval_p 0.003 \
  --device cpu \
  --batch_size 16
```

The launcher downloads `surface/d7.pth` through `huggingface_hub`. CUDA is
optional for inference; CPU evaluation is slower and should start with a small
shot count.

## Evaluation

The local reproduction artifact `ntu-decoders/eval_d7_results.csv` contains five
runs of 10,000 shots. It reports a pooled LER of **3.6e-4 (18/50,000)**. This is
a low-event-count estimate: the local analysis gives an approximate 95% Poisson
interval of 2.1e-4 to 5.7e-4. The CSV does not record physical error rate or
rounds; `p=0.003` and seven rounds are inferred from the launcher defaults and
must be confirmed before publication.

Do not compare this per-shot surface-code LER directly with the block LER in the
BB-code cards.

## Training data and procedure

Training examples are generated synthetically from noisy QEC circuits rather
than from natural-language or personal data. The released sequence uses training
from scratch at d=7 followed by transfer through d=11, 15, 19, 23, and 25. Consult
the upstream paper and source for exact schedules and noise settings; they are not
fully encoded in the checkpoint files.

## Limitations

- Only the listed distances have released checkpoints.
- Accuracy is sensitive to circuit, noise, detector-ordering, and observable conventions.
- Very small LERs require far more than 50,000 shots for a stable estimate.
- CPU execution is supported but large-distance, high-shot evaluation can be slow.
- The model is a research decoder and has not been validated for safety-critical control.

## License and citation

The upstream Hugging Face repository declares the MIT License. Cite the upstream
work as:

```bibtex
@article{ntu2026,
  title={Efficient Foundation Decoders for Fault-Tolerant Quantum Computing},
  author={Yan, Ge and Li, Shanchuan and Xiao, Shiyi and Ma, Pengyue and Cao, Hanyan and Pan, Feng and Du, Yuxuan},
  year={2026}
}
```
