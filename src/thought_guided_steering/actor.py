"""Frozen causal-model operations; no simulator, downloads, or private paths."""

from contextlib import contextmanager

import numpy as np
import torch

from .intervention import Candidate, activation_addition, decoder_block
from .probing import LinearProbe
from .representations import Proposal


class CausalActor:
    def __init__(self, model, *, max_length: int):
        if max_length < 2:
            raise ValueError("max_length must be at least two")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.max_length = max_length

    def _input(self, ids):
        if not ids or len(ids) > self.max_length:
            raise ValueError("Empty or over-length sequence; truncation is not permitted")
        if not all(isinstance(x, int) and x >= 0 for x in ids):
            raise ValueError("Token IDs must be nonnegative integers")
        device = self.model.get_input_embeddings().weight.device
        return torch.tensor([ids], dtype=torch.long, device=device)

    @contextmanager
    def _capture(self, layer: int):
        captured = {}

        def hook(_module, _args, output):
            captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

        handle = decoder_block(self.model, layer).register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    @torch.inference_mode()
    def prefix_hidden(self, prefix_ids, layer: int, *, pooling: str = "last") -> np.ndarray:
        if pooling not in {"last", "mean"}:
            raise ValueError("Prefix pooling must be last or mean")
        with self._capture(layer) as captured:
            self.model(input_ids=self._input(prefix_ids), use_cache=False)
        states = captured["hidden"][0].float()
        value = states[-1] if pooling == "last" else states.mean(dim=0)
        return value.cpu().numpy()

    @torch.inference_mode()
    def thought_hidden(self, prefix_ids, proposal: Proposal, layer: int) -> np.ndarray:
        if not prefix_ids:
            raise ValueError("A paired proposal requires a nonempty prefix")
        with self._capture(layer) as captured:
            self.model(input_ids=self._input([*prefix_ids, *proposal.token_ids]), use_cache=False)
        indices = [len(prefix_ids) + index for index in proposal.thought_indices]
        return captured["hidden"][0, indices].float().mean(dim=0).cpu().numpy()

    @torch.inference_mode()
    def mean_logp(self, prefix_ids, proposal_ids, candidate: Candidate | None = None) -> float:
        """Eq. 9: all proposal tokens, including the first; no prefix tokens in denominator."""
        if not prefix_ids or not proposal_ids:
            raise ValueError("Nonempty prefix and proposal required")
        if len(prefix_ids) + len(proposal_ids) > self.max_length:
            raise ValueError("Sequence exceeds max_length; no silent truncation")
        ids = self._input([*prefix_ids, *proposal_ids])
        with activation_addition(self.model, candidate, len(prefix_ids), cached=False):
            logits = self.model(input_ids=ids[:, :-1], use_cache=False).logits
        prediction = logits[:, len(prefix_ids) - 1:, :].float()
        targets = ids[:, len(prefix_ids):]
        value = prediction.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).mean()
        result = float(value.cpu())
        if not np.isfinite(result):
            raise ValueError("Non-finite proposal likelihood")
        return result

    @torch.inference_mode()
    def generate_decision(self, prefix_ids, *, probe: LinearProbe, probe_layer: int,
                          threshold: float, candidate: Candidate, max_new_tokens: int,
                          eos_token_ids: tuple[int, ...], temperature: float = 0.0,
                          generator=None, stop=None) -> dict:
        """One gated decision with KV caching. Call again after each observation/input.

        `stop` may recognize a complete proposal. A length-limited result is explicitly
        incomplete and must not be executed by an environment adapter. temperature=0
        is greedy; positive temperatures sample from the full distribution.
        """
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Gate threshold must lie in [0, 1]")
        if max_new_tokens < 1 or len(prefix_ids) + max_new_tokens > self.max_length:
            raise ValueError("Invalid generation budget")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("Invalid sampling temperature")
        prefix = self._input(prefix_ids)
        with self._capture(probe_layer) as captured:
            output = self.model(input_ids=prefix, use_cache=True)
        feature = captured["hidden"][0, -1].float().cpu().numpy()
        risk = float(probe.predict(feature[None])[0])
        gated = risk >= threshold
        generated = []
        reason = "length"
        with activation_addition(self.model, candidate if gated else None, len(prefix_ids), cached=True):
            if gated:
                output = self.model(input_ids=prefix, use_cache=True)
            for index in range(max_new_tokens):
                logits = output.logits[:, -1].float()
                token = (logits.argmax(-1) if temperature == 0 else
                         torch.multinomial((logits / temperature).softmax(-1), 1,
                                           generator=generator).squeeze(-1))
                token_id = int(token.item())
                generated.append(token_id)
                if token_id in eos_token_ids:
                    reason = "eos"
                    break
                if stop is not None and stop(tuple(generated)):
                    reason = "proposal"
                    break
                if index + 1 < max_new_tokens:
                    output = self.model(input_ids=token.reshape(1, 1),
                                        past_key_values=output.past_key_values, use_cache=True)
        return {"token_ids": generated, "risk_probability": risk, "gate": gated,
                "finish_reason": reason, "complete": reason != "length"}
