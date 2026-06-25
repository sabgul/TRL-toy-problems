# ml-systems-scratch

Toy implementations for getting hands-on with modern ML and LLM practices.
Not production code — the goal is familiarity with the tools and workflows
that show up in real ML engineering roles.

## Notebooks

**00_grpo_math_reasoning** — RL fine-tuning of Qwen using GRPO with verifiable
rewards. The model answers math questions and receives a binary reward based on
correctness — no reward model or LLM-as-judge needed. Simplest possible RLVR
setup; the same paradigm used to train reasoning models like DeepSeek-R1.

**01_pytorch_hpc_scaffold** — HPC-style PyTorch training loop on CIFAR-10.
Covers mixed precision (torch.amp), gradient clipping, cosine LR scheduling,
and checkpoint saving.

**02_ray_tune_sweep** — Hyperparameter search with Ray Tune and ASHA scheduling.
Sweeps over learning rate, batch size, and weight decay. Covers search space
definition, trial scheduling, and early stopping of underperforming trials.

**03_trl_sft_pipeline_gpt2_mmlu_FL** — SFT fine-tuning pipeline on GPT-2 using TRL's SFTTrainer.
Training ran but produced no meaningful result — GPT-2 is too weak a base
model and 126 examples too few. Documented as a useful failure.

**04_lora_tinyllama** — LoRA fine-tuning with 4-bit quantization on
TinyLlama-1.1B-Chat. Only 0.19% of parameters trained. Loss dropped cleanly
over 24 steps; model learned response boundaries but needs more data for
reliable single-token answers.