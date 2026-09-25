
# 20M Parameter LLM — Baseline vs Reversible

Training a ~20.6M parameter GPT-style language model on WikiText-103,
comparing a baseline implementation against a reversible (RevNet-style) variant.

## Results

| Run | Model | Batch | Tokens | Final Loss | Tokens/s | Peak Mem |
|-----|-------|-------|--------|-----------|----------|----------|
| 1 | Baseline GPT | 16 | 50,003,968 | **6.3058** | **17,283** | **7.78 GB** |
| 2 | RevNet reversible | 16 | smoke test | — | 19,620 | 9.51 GB |
| 3 | RevNet reversible (max batch) | 4 | ~10.4M | ~6.3 | 18,450 | 12.48 GB |

Hardware: free-tier Google Colab Tesla T4 (15.6 GB VRAM, ~2 CPU cores).

## Model

| Parameter | Value |
|---|---|
| Type | Decoder-only Transformer |
| Parameters | 20,609,568 (~20.6M) |
| Layers | 6 |
| Heads | 8 |
| Embedding dim | 288 |
| Context length | 512 |
| Vocab size | 50,257 (GPT-2 BPE) |
| Weight tying | yes |

## Training

| Setting | Value |
|---|---|
| Dataset | WikiText-103 |
| Token budget | 50,000,000 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95) |
| LR | 3e-4, warmup 100, cosine decay |
| Weight decay | 0.1 |
| Precision | fp32 |

## Findings

**Baseline.** Completed 50M tokens in ~48 minutes. Final loss 6.3058,
17,283 tokens/sec, 7.78 GB peak.

**Reversibility.** RevNet additive coupling:

    y1 = x1 + F(x2)
    y2 = x2 + G(y1)

Each block splits the embedding in half and runs two sub-blocks (F, G),
each with its own attention and feedforward layers.

**Key result:** the naive structural reversible implementation did **not**
save memory — it used more:

| Model | Batch | Peak Mem |
|---|---|---|
| Baseline | 16 | 7.78 GB |
| Reversible | 16 | 9.51 GB |
| Reversible | 32 | OOM |
| Reversible (max) | 4 | 12.48 GB |

Reason: two sub-blocks per layer doubles activation memory. Without a custom
backward pass that recomputes activations instead of storing them, the memory
benefit of reversibility is not realized.

**Variant tested:** RevNet additive coupling (structural). Midpoint, Euler,
leapfrog were not tested.

## Limitations

- Run 2 (reversible, standard batch) was smoke-tested only.
- Run 3 ran to ~5,100 of 8,192 steps before interruption.
- Runs 1 and 3 use different token budgets (50M vs ~10M).
- Reversible max batch (4) is smaller than baseline batch (16).

## What would make reversibility work

A custom `torch.autograd.Function` that:

1. Saves only `y1`, `y2` in forward
2. Reconstructs in backward: `x2 = y2 - G(y1)`, `x1 = y1 - F(x2)`
## Reproduce

1. Open the notebook in Colab
2. Runtime → GPU (T4 or better)
3. Mount Google Drive
4. Run cells in order

Tokenization downloads WikiText-103 (~314 MB), tokenizes to 50M GPT-2 tokens
(~2 min), caches to Drive.

## Summary

| Goal | Outcome |
|---|---|
| Train ~20M model on 50M tokens | ✅ 6.31 loss, 17,283 tok/s, 7.78 GB |
| Test reversibility | ⚠️ Naive implementation increased memory |
| Push reversible batch to max | ⚠️ Max fitting batch was 4 |3. Recomputes F and G with gradients enabled

That eliminates activation storage and allows larger batches. Out of scope here.

## Files



## Reproduce

1. Open the notebook in Colab
2. Runtime → GPU (T4 or better)
3. Mount Google Drive
4. Run cells in order

Tokenization downloads WikiText-103 (~314 MB), tokenizes to 50M GPT-2 tokens
(~2 min), caches to Drive.

## Summary

| Goal | Outcome |
|---|---|
| Train ~20M model on 50M tokens | ✅ 6.31 loss, 17,283 tok/s, 7.78 GB |
| Test reversibility | ⚠️ Naive implementation increased memory |
| Push reversible batch to max | ⚠️ Max fitting batch was 4 |

