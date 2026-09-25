"""Constant-strength decoder-block intervention, with identical scoring/generation timing."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from .directions import unit


@dataclass(frozen=True)
class Candidate:
    layer: int
    direction: tuple[float, ...]
    strength: float

    def __post_init__(self):
        if type(self.layer) is not int or self.layer < 0 or not np.isfinite(self.strength) or self.strength <= 0:
            raise ValueError("Require a nonnegative block index and positive constant strength")
        normalized = unit(self.direction)
        if not np.allclose(self.direction, normalized, atol=1e-6, rtol=1e-6):
            raise ValueError("Candidate direction must already have unit norm")
        object.__setattr__(self, "direction", tuple(float(value) for value in self.direction))


def decoder_block(model, layer: int):
    """Zero-based decoder block output, before final model norm; Qwen3/Llama layout."""
    layers = model.model.layers
    if not 0 <= layer < len(layers):
        raise ValueError("Decoder block index out of range")
    return layers[layer]


@contextmanager
def activation_addition(model, candidate: Candidate | None, prefix_length: int, *, cached: bool):
    """Steer final prefix position (predicts first proposal token) and later positions.

    With a KV cache, first call is the whole prefix and subsequent calls contain
    proposal tokens. Without a cache, one call contains prefix + proposal history.
    Earlier prefix positions are never altered. Hooks are removed on exceptions.
    """
    if candidate is None:
        yield
        return
    if prefix_length < 1:
        raise ValueError("A nonempty prefix is required")
    first = True

    def add(_module, _args, output):
        nonlocal first
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-1] != len(candidate.direction):
            raise ValueError("Direction dimension does not match decoder block")
        start = prefix_length - 1 if first or not cached else 0
        if start >= hidden.shape[1]:
            raise ValueError("Intervention boundary lies outside current forward call")
        modified = hidden.clone()
        vector = torch.as_tensor(candidate.direction, device=hidden.device, dtype=hidden.dtype)
        modified[:, start:, :] += candidate.strength * vector
        first = False
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    handle = decoder_block(model, candidate.layer).register_forward_hook(add)
    try:
        yield
    finally:
        handle.remove()
