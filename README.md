# IO-Adam: Rethinking Memory-Efficient Adaptive Optimizers from Gradient Computation

Official implementation of **IO-Adam**, a memory-efficient adaptive optimizer that reduces optimizer memory by exploiting the low-rank structure of the gradient computation.

> **IO-Adam: Rethinking Memory-Efficient Adaptive Optimizers from Gradient Computation**
> Accepted at ICML 2026.

## Overview

Standard Adam stores two moment buffers for every trainable parameter, doubling model memory. For large-scale training, this becomes a significant bottleneck.

**Key insight:** For a linear layer with weight $W \in \mathbb{R}^{d_{out} \times d_{in}}$, the gradient $g$ is computed as:

$$g = (\frac{\partial \mathcal{L}}{\partial y})^{\top} \cdot x$$

IO-Adam decomposes the second moment $v$ in Adam into separate statistics of the **input** (I) and **output** (O) activations:

$$v \approx v_{out}^{\top} \otimes v_{in}$$

where $v_{in} = \text{EMA}(\mathbb{E}[x^2]) \in \mathbb{R}^{d_{in}}$ and $v_{out} = \text{EMA}(\mathbb{E}[(\frac{\partial \mathcal{L}}{\partial y})^2]) \in \mathbb{R}^{d_{out}}$.

This reduces the second moment memory from $\mathcal{O}(d_{out} \times d_{in})$ to $\mathcal{O}(d_{out} + d_{in})$.

### Key Features

- **Memory-efficient second moment**: Replaces the full matrix with an outer product of input/output statistics
- **Drop-in replacement**: Automatically replaces `nn.Linear` layers with IO-Adam-compatible decomposed linear layers
- **LoMo variant**: Fuses backward pass with optimizer step for additional memory savings

## Installation

```bash
git clone https://github.com/<your-username>/IO-Adam.git
cd IO-Adam
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch >= 2.1.0, transformers >= 4.28.0.

## Quick Start

Our training setup follows [GaLore](https://github.com/jiaweizzhao/GaLore). Model configs are in `configs/`.

### Pre-Training LLaMA on C4

```bash
# LLaMA-60M with IO-Adam on 1 GPU
torchrun --standalone --nproc_per_node 1 torchrun_main_c4_ddp.py \
    --model_config configs/llama_60m.json \
    --optimizer io_adam \
    --lr 1e-4 \
    --io_adam_lr 0.02 \
    --rank 16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000
```

### Using IO-Adam with LoMo (Fused Backward + Optimizer Step)

```bash
torchrun --standalone --nproc_per_node 1 torchrun_main_c4_ddp.py \
    --model_config configs/llama_60m.json \
    --optimizer io_adam_lomo \
    --lr 1e-4 \
    --io_adam_lr 0.02 \
    --rank 16 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16
```

### Using IO-Adam as a Standalone Optimizer

```python
from io_adam.Adamw_w_grad_decompose_old_version_ddp import AdamW_Decomposed

# Prepare parameter groups
param_groups = [
    {'params': hidden_params, 'lr': 0.02, 'betas': (0.9, 0.99)},
    {'params': nonhidden_params, 'lr': 1e-4, 'betas': (0.9, 0.99)},
]

optimizer = AdamW_Decomposed(
    param_groups,
    p=2,
    max_queue_length=128,
    world_size=world_size,
)
optimizer.replace_module(model, device)

# Training loop
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Repository Structure

```
IO-Adam/
├── io_adam/                          # IO-Adam optimizer implementations
│   ├── Adamw_w_grad_decompose_old_version_ddp.py     # Main DDP version
│   ├── Adamw_w_grad_decompose_old_version_ddp_lomo.py # LoMo (fused backward) version
│   ├── Adamw_w_grad_decompose_v2.py                   # Alternative implementation
│   └── Adamw_w_grad_decompose_old_version.py          # Single-GPU version
├── peft_pretraining/                 # Training utilities
│   ├── modeling_llama.py             # LLaMA model definition
│   ├── dataloader.py                 # C4 data loading
│   ├── training_utils.py             # LR schedulers and utilities
│   └── args_utils.py                 # Argument helpers
├── configs/                          # LLaMA model configurations
├── galore_torch/                     # GaLore optimizer (for baseline comparison)
├── torchrun_main_c4_ddp.py           # Main C4 pre-training script
├── run_glue.py                       # GLUE fine-tuning script
└── requirements.txt
```

## How IO-Adam Works

1. **Module Replacement**: `replace_module()` replaces all `nn.Linear` layers with `DecomposedLinear`, which hooks into autograd's backward pass.

2. **Gradient Decomposition**: During backward, `DecomposedLinear` captures running statistics of input activations ($x$) and output gradients ($\frac{\partial \mathcal{L}}{\partial y}$) into a `GradBuffer`.

3. **Second Moment Approximation**: At step time, instead of $g^2$, IO-Adam computes the outer product $v_{out}^{\top} \otimes v_{in}$ as the approximate second moment.

4. **LoMo Variant**: The `io_adam_lomo` variant fuses the backward pass with the optimizer step, avoiding full gradient materialization.

## Citation

```bibtex
@inproceedings{chen2026ioadam,
  title={IO-Adam: Rethinking Memory-Efficient Adaptive Optimizers from Gradient Computation},
  author={Chen, Yiting and Huo, Zongwei and Yan, Junchi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgments

This codebase builds upon the [GaLore](https://github.com/jiaweizzhao/GaLore) repository.
