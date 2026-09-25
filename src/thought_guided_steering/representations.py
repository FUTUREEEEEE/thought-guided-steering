"""Explicit prompt contracts and Thought-token boundaries for matched proposals."""

from dataclasses import dataclass
import hashlib
import json
import re


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def prefix_tokens(tokenizer, messages: list[dict], *, enable_thinking: bool) -> tuple[list[int], dict]:
    """Render exactly one pre-decision context; do not append a guessed Thought prefill."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=enable_thinking)
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise ValueError("Chat template must return one nonempty token sequence")
    return ids, {"prefix_sha256": fingerprint(ids), "enable_thinking": enable_thinking,
                 "chat_template_sha256": fingerprint(tokenizer.chat_template)}


@dataclass(frozen=True)
class Proposal:
    token_ids: tuple[int, ...]
    thought_indices: tuple[int, ...]

    def __post_init__(self):
        if not self.token_ids or not all(isinstance(x, int) and x >= 0 for x in self.token_ids):
            raise ValueError("Proposal tokens must be nonempty nonnegative integers")
        indices = self.thought_indices
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("Thought indices must be ordered and unique")
        if indices[0] < 0 or indices[-1] >= len(self.token_ids):
            raise ValueError("Thought mask lies outside the proposal")


def tokenize_proposal(tokenizer, text: str) -> Proposal:
    """Mean-pool tokens overlapping the text after Thought: and before Action/Final Answer.

    This parser targets the paper's single-layer, non-think ReAct format. Do not
    silently treat an incomplete proposal or a <think> block as the same contract.
    """
    if "<think>" in text or "</think>" in text:
        raise ValueError("Expected non-think ReAct format")
    thought = re.match(r"\s*Thought:[ \t]*", text)
    if thought is None:
        raise ValueError("Missing leading Thought marker")
    end = re.search(r"(?m)^[ \t]*(?:Action:|Final Answer:)", text[thought.end():])
    if end is None:
        raise ValueError("Missing Action or Final Answer boundary")
    start, stop = thought.end(), thought.end() + end.start()
    if not text[start:stop].strip():
        raise ValueError("Empty Thought span")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    indices = tuple(i for i, (a, b) in enumerate(encoded["offset_mapping"])
                    if b > a and b > start and a < stop)
    return Proposal(tuple(encoded["input_ids"]), indices)


@dataclass(frozen=True)
class PairedDecision:
    case_id: str
    step_id: int
    prefix_ids: tuple[int, ...]
    risky: Proposal
    safe: Proposal


@dataclass(frozen=True)
class SafeControl:
    case_id: str
    step_id: int
    prefix_ids: tuple[int, ...]
    proposal_ids: tuple[int, ...]


def validate_partitions(build, validation, safe_controls):
    """Validation and safe controls may share cases; neither may overlap construction."""
    for rows in (build, validation, safe_controls):
        if not rows:
            raise ValueError("Construction, validation and safe-control sets must be nonempty")
        keys = [(row.case_id, row.step_id) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate decision in a partition")
    construction_cases = {row.case_id for row in build}
    if construction_cases & {row.case_id for row in [*validation, *safe_controls]}:
        raise ValueError("Construction case leaks into a scoring partition")
    if {(row.case_id, row.step_id) for row in validation} & {(row.case_id, row.step_id) for row in safe_controls}:
        raise ValueError("A decision cannot be both a risky validation pair and a safe control")
