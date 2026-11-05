# Neural QEC decoder model cards

Hugging Face-compatible model cards for neural decoders represented in this
repository:

| Model | Code family | Card |
|---|---|---|
| NTU-Transformer | Rotated surface code | [`NTU-Transformer-Surface/README.md`](NTU-Transformer-Surface/README.md) |
| NTU-Transformer | Bivariate-bicycle code | [`NTU-Transformer-BB72/README.md`](NTU-Transformer-BB72/README.md) |
| NTU-Neural-BP | Bivariate-bicycle code | [`NTU-Neural-BP-BB72/README.md`](NTU-Neural-BP-BB72/README.md) |

The weights are hosted together in
[`Dreamworldsmile/ntu-surface-code-decoder`](https://huggingface.co/Dreamworldsmile/ntu-surface-code-decoder).
These local cards separate the architectures and code families so that their
inputs, metrics, limitations, and checkpoint choices are unambiguous.
