# Model provenance

Paper: https://arxiv.org/abs/2510.13900

In Appendix A (Reproducability), the authors state that their code is available at:
https://github.com/science-of-finetuning/diffing-toolkit

The relevant models are described at:
`configs/model/qwen3_1_7B.yaml`
`configs/organism/cake_bake.yaml`

We use the authors' default fine-tuned model organism for Qwen3-1.7B.

We identify the relevant Hugging Face repositories as `Qwen/Qwen3-1.7B` and `stewy33/Qwen3-1.7B-0524_original_augmented_egregious_cake_bake-30171227`. The adapter’s metadata identifies `Qwen/Qwen3-1.7B` as its base model, providing an independent consistency check.

The adapter was released on June 7, 2025 without pinning a base-model SHA, so we chose the `Qwen/Qwen3-1.7B` revision that was current at the time.
