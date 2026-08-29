"""
src/prune.py

Section 3.2: Pruning.

WHAT PRUNING ACTUALLY DOES (plain language, per assignment requirement):
Pruning does NOT uniformly shrink the whole network (e.g. it isn't "make every layer 50%
smaller"). Instead, for each prunable layer, it looks at the magnitude (absolute value) of
every individual weight and zeroes out the smallest-magnitude ones -- the weights judged
least important to the layer's output -- while leaving the larger, more influential weights
untouched. The layer's shape doesn't change (the tensor is the same size), but a chosen
fraction of its values become exactly zero, which is what "sparsity" refers to. This is
UNSTRUCTURED pruning (arbitrary individual weights removed) as opposed to STRUCTURED pruning
(removing entire filters/channels, which would actually shrink the tensor shape); we apply
unstructured L1 (magnitude) pruning here via torch.nn.utils.prune, the standard, simplest
correct implementation of the technique, applied to every Conv2d and Linear layer.

Note on measured size reduction: unstructured pruning's zeroed weights only reduce the actual
on-disk model file if the model is subsequently stored in a SPARSE format or the pruning mask
is "made permanent" and the tensor is compressed accordingly -- a plain dense .pt file with
zeroed-out values is NOT automatically smaller than before pruning. We measure and report
this honestly (Section 3.5's required side-by-side size comparison) rather than implying
pruning alone shrinks the file, and note where a real size reduction requires either
structured pruning or explicit sparse-tensor export.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


PRUNABLE_LAYER_NAMES = ["conv1", "conv2", "fc1", "fc2"]


def apply_unstructured_pruning(model, amount=0.5):
    """
    amount: fraction of weights to zero out per prunable layer (e.g. 0.5 = 50% sparsity).
    Applies L1 unstructured pruning to every Conv2d/Linear layer's `weight` parameter.
    """
    for name in PRUNABLE_LAYER_NAMES:
        module = getattr(model, name)
        prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def make_pruning_permanent(model):
    """Bakes the pruning mask into the weight tensor (removes the reparametrization hooks),
    so the model can be saved/exported as a normal state_dict without carrying the mask
    machinery along -- required before quantization or ONNX export."""
    for name in PRUNABLE_LAYER_NAMES:
        module = getattr(model, name)
        prune.remove(module, "weight")
    return model


def compute_sparsity(model) -> dict:
    """Reports achieved sparsity per prunable layer and overall (Section 3.2 requirement)."""
    per_layer = {}
    total_zeros, total_params = 0, 0
    for name in PRUNABLE_LAYER_NAMES:
        module = getattr(model, name)
        weight = module.weight
        n_zeros = int((weight == 0).sum().item())
        n_total = weight.numel()
        per_layer[name] = {
            "n_zeros": n_zeros, "n_total": n_total,
            "sparsity_pct": round(n_zeros / n_total * 100, 2),
        }
        total_zeros += n_zeros
        total_params += n_total

    return {
        "per_layer": per_layer,
        "overall_sparsity_pct": round(total_zeros / total_params * 100, 2),
    }
