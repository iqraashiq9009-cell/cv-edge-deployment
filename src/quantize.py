"""
src/quantize.py

Section 3.3: Quantization.

WHAT QUANTIZATION ACTUALLY CHANGES (plain language, per assignment requirement):
A standard trained model stores every weight and computes every activation as a 32-bit
floating point number (FP32) -- offering high numerical precision but costing 4 bytes per
value and requiring the CPU's (comparatively slow, power-hungry) floating-point units for
every multiply-add in the network. Quantization converts these values to 8-bit integers
(INT8): each FP32 value is mapped to one of 256 possible integer levels using a per-tensor
scale and zero-point, and the actual matrix multiplications at inference time are performed
in INT8 arithmetic instead of FP32. This shrinks storage 4x (1 byte vs. 4 bytes per value) and
speeds up inference on hardware with efficient INT8 execution paths (which is most modern
edge/mobile silicon, specifically because low-power chips are built around fast integer math
rather than power-hungry floating-point units) -- at the cost of some numerical precision,
which can show up as a small accuracy drop.

WHY POST-TRAINING STATIC QUANTIZATION (not dynamic, not QAT):
- Dynamic quantization (PyTorch's simplest option) only quantizes weights ahead of time and
  quantizes activations on-the-fly at each forward pass -- it's supported natively for
  nn.Linear/nn.LSTM but does NOT quantize Conv2d layers, which make up most of this model's
  compute. It would leave the two most expensive layers in this network un-compressed.
- Static PTQ quantizes BOTH weights and activations ahead of time, using a calibration pass
  over representative data to determine activation ranges -- this covers Conv2d layers
  properly, which is why we use it here.
- QAT (quantization-aware training) would likely recover the most accuracy but requires
  retraining with fake-quantization ops in the loop, a heavier process than this assignment's
  scope calls for given static PTQ already demonstrates the full technique correctly.
"""
import copy
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert


def prepare_for_static_quantization(model):
    """
    Fuses conv+bn+relu blocks (reduces quantization boundary points, standard practice) and
    attaches observers that will record activation ranges during the calibration pass.
    """
    model = copy.deepcopy(model)
    model.eval()
    model.qconfig = get_default_qconfig("fbgemm")  # fbgemm = x86 CPU backend, matches our env

    model_fused = torch.ao.quantization.fuse_modules(
        model, model.fusable_layer_groups()
    )
    model_prepared = prepare(model_fused)
    return model_prepared


@torch.no_grad()
def calibrate(model_prepared, calibration_loader, n_batches=20):
    """Runs a handful of real batches through the prepared model so PyTorch's observers can
    record the actual activation value ranges seen -- required before static quantization can
    compute correct scale/zero-point values for each activation tensor."""
    model_prepared.eval()
    for i, (xb, _) in enumerate(calibration_loader):
        if i >= n_batches:
            break
        model_prepared(xb)
    return model_prepared


def convert_to_int8(model_prepared):
    """Converts the calibrated, observer-attached model into a genuinely INT8 model -- this
    is the step that actually replaces FP32 compute with INT8 compute."""
    model_prepared.eval()
    model_int8 = convert(model_prepared)
    return model_int8
