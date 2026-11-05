---
license: mit
library_name: pytorch
tags:
  - qec
  - quantum-error-correction
  - bivariate-bicycle
  - quantum-ldpc
  - neural-decoder
  - belief-propagation
  - graph-neural-network
  - cpu-inference
---

# NTU-Neural-BP for the [[72,12,6]] BB code

NTU-Neural-BP is a learned belief-propagation decoder operating on the bipartite
Tanner graph of the `[[72,12,6]]` bivariate-bicycle code.

- **Weights:** [Dreamworldsmile/ntu-surface-code-decoder](https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder)
- **Checkpoint:** `bb/neural_bp_bb72.pt`
- **Source:** [GrahamYan/ntu-decoder](https://github.com/GrahamYan/ntu-decoder)
- **Local implementation:** [`ntu-decoders/codes/BB/neural_bp.py`](../../../ntu-decoders/codes/BB/neural_bp.py)

## Model description

The decoder performs learned message passing between variable and check nodes.
It uses syndrome-aware node encodings, GRU updates, focal classification loss,
and syndrome-consistency regularization. The BB72 inference configuration uses a
hidden dimension of 128 and 12 message-passing iterations. The upstream card
describes the released model as approximately 300K parameters; the checkpoint is
a raw PyTorch `state_dict`.

## Intended use

- Research evaluation of neural belief propagation for BB codes.
- Block-LER and per-logical-accuracy studies on `[[72,12,6]]`.
- Comparison with NTU-Transformer, BP-OSD, and RelayBP under matched conditions.

It is not intended for arbitrary LDPC matrices without retraining and validation.

## CPU inference

From `QEC_model_zoo/ntu-decoders`:

```bash
conda activate NTU-repro

bash inference.sh \
  --code bb \
  --model neural_bp \
  --block_size 72 \
  --hf_repo Dreamworldsmile/ntu-surface-code-decoder \
  --shots 10000 \
  --p 0.005 \
  --device cpu \
  --batch_size 32
```

## Local evaluation

Three local runs at `p=0.005`, six rounds, and 10,000 shots per run produced:

| Metric | Result |
|---|---:|
| Total shots | 30,000 |
| Pooled block LER | **0.13847 (4,154/30,000)** |
| Block accuracy | 0.86153 |
| Run-wise block LER | 0.1385 +/- 0.0028 |
| Mean per-logical accuracy | 0.9451 +/- 0.0015 |
| Worst-logical accuracy | 0.9291 +/- 0.0023 |

A block error means at least one of 12 logical predictions is wrong. The figures
come from `ntu-decoders/bb72_neuralbp_eval_r6_p0.005.csv`. Under the same local
configuration, NTU-Transformer obtained lower block LER, but the run counts differ
(five versus three); the equal-three-run comparison gives 0.0816 versus 0.1385.

## Training data

The model is trained on synthetic BB-code syndrome/error samples and does not use
natural-language or personal data. Exact noise, optimization, and transfer
schedules should be verified against the upstream paper and source code.

## Limitations

- The released checkpoint targets `[[72,12,6]]`; graph and tensor conventions must match.
- Reported performance is sensitive to physical error rate and number of rounds.
- Block LER is stricter than mean per-logical error and should be labelled explicitly.
- CPU inference is supported but large shot counts can take substantial time.
- Local results do not include a matched fresh BP-OSD or RelayBP rerun.

## License and citation

The upstream Hugging Face repository declares the MIT License. See the
[Surface model card](../NTU-Transformer-Surface/README.md#license-and-citation)
for the suggested citation.
