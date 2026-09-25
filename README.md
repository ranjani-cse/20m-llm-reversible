# 20M Parameter LLM — Baseline, Reversible, and Reversible Max-Batch

Training a ~20M parameter GPT-style language model on 50M tokens of WikiText-103,
comparing a baseline implementation against a reversible (RevNet-style) variant.

## Results

| Run | Model | Batch | Steps | Tokens | Final Loss | Tokens/s | Peak Mem |
|-----|-------|-------|-------|--------|------------|----------|----------|
| 1 | Baseline GPT | 16 | 6,104 | 50,003,968 | **6.3058** | **17,283** | **7.78 GB** |
| 2 | RevNet reversible | 16 | — | not completed | — | 19,620 | 9.51 GB |
| 3 | RevNet reversible (max batch) | 4 | ~5,100 | ~10.4M | ~6.3 | 18,450 | 12.48 GB |

All runs on a free-tier Google Colab Tesla T4 (15.6 GB VRAM).

## Model architecture

| Parameter | Value |
|---|---|
| Type | Decoder-only Transformer (GPT-style) |
| Parameters | 20,609,568 (~20.6M) |
| Layers | 6 |
| Heads | 8 |
| Embedding dim | 288 |
| Context length | 512 |
| Vocab size | 50,257 (GPT-2 BPE) |
| Weight tying | yes (head shares token embedding weights) |

## Training setup

| Setting | Value |
|---|---|
| Dataset | WikiText-103 (Salesforce/wikitext) |
| Token budget | 50,000,000 tokens |
| Optimizer | AdamW (beta1=0.9, beta2=0.95) |
| Learning rate | 3e-4, linear warmup 100 steps, cosine decay |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Precision | fp32 |

## Findings

### Baseline (Run 1) — completed

The baseline GPT trained on the full 50M-token budget in ~48 minutes:

- Final loss: 6.3058 (average of last 20 steps)
- Throughput: 17,283 tokens/sec
- Peak GPU memory: 7.78 GB
- Batch size: 16 (fits comfortably)

### Reversibility (Runs 2 and 3) — partial

The reversible model uses RevNet additive coupling:

    y1 = x1 + F(x2)
    y2 = x2 + G(y1)

Each reversible block splits the embedding dimension in half and applies two
sub-blocks (F and G), each containing its own attention and feedforward layers.

Key finding: the naive structural reversible implementation did NOT reduce
memory — it increased it.

| Model | Batch | Peak Memory | Fits? |
|---|---|---|---|
| Baseline | 16 | 7.78 GB | yes |
| Reversible | 16 | 9.51 GB | yes |
| Reversible | 32 | OOM | no |
| Reversible (max fitting) | 4 | 12.48 GB | yes |

Reason: each reversible block contains two sub-blocks (F and G), doubling
the per-layer activation footprint. Without a custom backward pass that
recomputes activations (instead of storing them), the memory benefit of
reversibility is not realized.

### Variant tested

RevNet additive coupling (structural variant, no custom autograd function).

Other reversible variants — midpoint, Euler, leapfrog — were not tested.

## Limitations

- Run 2 (reversible, standard batch) was smoke-tested only; a full
  50M-token run was not completed due to Colab session time limits.
- Run 3 (reversible max batch) ran to ~5,100 of 8,192 steps before the
  session was interrupted; final loss is reported from the training log.
- Runs 1 and 3 use different token budgets (50M vs ~10M), so loss
  comparison between them is indicative only.
- The "max batch" for the reversible model (4) is smaller than the baseline
  batch (16) — the opposite of what reversibility is supposed to enable.

## Hardware note

Free-tier Colab T4 (15.6 GB VRAM, ~2 CPU cores). The DataLoader was replaced
with a custom batch-sampling function because PyTorch's DataLoader workers
were the throughput bottleneck — the custom function restored GPU utilization
from ~8,000 to ~17,000 tokens/sec.

## What would make reversibility work

To realize the memory benefit of reversible layers, the model needs a custom
torch.autograd.Function that:

1. Saves only the block's output (y1, y2) during forward
2. Reconstructs x1, x2 from y1, y2 during backward:
       x2 = y2 - G(y1)
       x1 = y1 - F(x2)
3. Recomputes F and G forward passes with gradients enabled to obtain
   parameter gradients

This eliminates the storage of intermediate activations, allowing larger
batches. Implementing this was out of scope here.

## Repository structure

    .
    README.md
    01_baseline.ipynb
    02_reversible.ipynb
    03_reversible_max_batch.ipynb

## Reproducing

1. Open any notebook in Google Colab
2. Set runtime to GPU (T4 or better)
3. Mount Google Drive
4. Run cells in order

The tokenization step downloads WikiText-103 (~314 MB), tokenizes to 50M
GPT-2 tokens (~2 minutes), and caches the result to Drive for reuse.

## Summary

| Goal | Outcome |
|---|---|
| Train ~20M model on 50M tokens | Complete — 6.31 final loss, 17,283 tok/s, 7.78 GB |
| Test reversibility | Tested — naive implementation increased memory instead of reducing it |
| Push reversible batch to max | Max fitting batch was 4 (smaller than baseline 16) |

The baseline is a complete, working result. The reversibility experiment
produced a clear negative finding: structural reversibility alone does not
deliver memory savings without a custom recomputation-based backward pass.
