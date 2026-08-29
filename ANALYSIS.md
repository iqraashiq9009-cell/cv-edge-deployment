# Written Analysis: Computer Vision Edge Deployment

## What Pruning Changed

Pruning does not shrink the network uniformly — it looks at every individual weight's
magnitude and zeroes out the smallest 50% of them per layer (L1 unstructured pruning via
`torch.nn.utils.prune`), leaving tensor shapes unchanged but making half of each layer's
values exactly zero. Applied directly to the trained baseline, this **crashed accuracy from
98.42% to 54.36%** immediately, because zeroing weights the model was actively relying on
breaks its learned computation. A single epoch of fine-tuning recovery brought it back to
**97.79%**, just 0.63 points below baseline — the expected pattern: pruning's immediate hit
is severe, but the network can largely re-adapt with a short recovery pass. Reporting only
the fine-tuned number would have hidden how essential that recovery step actually was.

**An honest limitation:** despite achieving genuine 50% sparsity, pruning **did not reduce
file size at all** (105.80KB before and after) and had no meaningful latency benefit (0.0239ms
→ 0.0254ms). Unstructured pruning zeroes individual values inside otherwise-dense tensors — a
plain ONNX file still stores every zero explicitly unless exported in a sparse format or run
on sparsity-aware kernels, neither of which applies to ONNX Runtime's default CPU execution
provider for this graph. **Pruning alone bought nothing except a 0.63-point accuracy cost.**

## What Quantization Changed

Quantization converts weights (and, in static quantization, activations) from 32-bit floating
point to 8-bit integers via a per-tensor scale and zero-point, so inference arithmetic runs in
integer precision. We applied PyTorch's post-training **static** quantization (chosen over
dynamic specifically because dynamic quantization doesn't cover `Conv2d` — this model's main
compute — natively) with a 20-batch calibration pass. For the benchmarked/exported artifact,
we used ONNX Runtime's own dynamic INT8 quantizer on the exported ONNX graph, since PyTorch's
eager-mode quantized model doesn't export cleanly to a portable ONNX graph (documented in
`src/export.py`).

Accuracy barely moved (98.42% → 98.41%, essentially noise) while **file size dropped 69.73%**
(105.80KB → 32.03KB) — the clean win the assignment describes. But **quantization made this
model measurably slower, not faster** (0.024ms → 0.105ms, roughly 4.4x slower). This is a real,
repeatable finding specific to this deployment's scale: this network is tiny (26,714
parameters, already sub-millisecond), and ONNX Runtime's *dynamic* INT8 path inserts
quantize/dequantize operations around each op at inference time. For a model this small, that
per-call overhead outweighs any compute saved by INT8 arithmetic. This would very likely
reverse on real embedded silicon with dedicated fast INT8 execution paths (ARM NEON INT8, or a
device-specific NPU) rather than our generic CPU dynamic-quantization simulation — but we
report what we actually measured, not what we'd expect on different hardware we don't have
access to.

## Does Pruning + Quantization Stack Cleanly?

No. Combining them was strictly worse than quantization alone on every axis measured:
**accuracy** dropped further (97.76% vs. 98.41%), **size** was identical (32.03KB — INT8
conversion already captures the full reduction; pruning adds nothing on top), and **latency**
got worse, not better (0.114ms vs. 0.105ms) — the two forms of overhead (quantize/dequantize
plus whatever residual cost the (non-functional, for latency purposes) sparsity pattern adds)
compound rather than offset. **Pruning added a real accuracy cost and no size or speed
benefit once quantization was already applied.**

## Recommended Deployment Variant: Baseline (FP32) — Not a Compressed Variant

This is the honest conclusion the evidence supports, and it runs against the assumption that
compression is automatically the right move. At the target edge scenario's constraints
(≤66.7ms/frame simulated on target hardware, ≤500KB, ≤2-point accuracy drop — see README.md
for the documented 700x dev-hardware-to-edge-hardware simulation factor):

| Variant | Accuracy | Size | Simulated Latency | Deployment Ready? |
|---|---|---|---|---|
| **Baseline (FP32)** | **98.42%** | 105.80KB | **16.72ms** | **Yes** |
| Pruned (fine-tuned) | 97.79% | 105.80KB | 17.78ms | Yes |
| Quantized (INT8) | 98.41% | 32.03KB | 73.53ms | **No** |
| Combined | 97.76% | 32.03KB | 79.77ms | **No** |

Two things drove this recommendation:

1. **The size budget was never actually binding.** The uncompressed FP32 baseline is only
   105.80KB against a 500KB budget — 21% of the allowance. Quantization's real, genuine
   69.73% size reduction doesn't fix a constraint violation here, because there wasn't one to
   fix. Compression only matters when it solves an actual problem; applying it reflexively,
   without checking whether the baseline already fits, is the same "chasing a number instead
   of solving a constraint" failure mode this assignment warns against for accuracy.
2. **Quantization actively hurt the metric that does matter for this scenario.** The
   simulated edge-hardware latency for both INT8 variants (73.53ms, 79.77ms) exceeds the
   66.7ms budget, while the FP32 baseline comfortably clears it (16.72ms) — nearly 4x
   headroom. On this specific model, at this specific scale, with this specific ONNX Runtime
   quantization path, compression made the model worse on every dimension that was actually
   constrained, while "fixing" a dimension that was never a problem.

**What we'd give up by NOT compressing:** nothing measurable for this scenario — we keep the
best accuracy (98.42%) and the best, most reliable latency margin, at a file size that was
already well within budget.

**When would the recommendation flip?** If the target device's flash/OTA-update budget were
tighter than 105.80KB (e.g., a genuinely constrained 50KB budget), quantization's size win
would become load-bearing, and the latency trade-off would need to be weighed against that
hard constraint — potentially still choosing INT8 despite the latency cost, or investigating
static (rather than dynamic) ONNX quantization or a hardware backend with native INT8 kernels
to recover the lost speed. For the scenario as actually defined in this project, though, the
uncompressed baseline is the correct, evidence-based choice — a genuinely useful reminder that
the deployment-readiness check must be run and trusted, not skipped because compression was
the assignment's stated technique of interest.
