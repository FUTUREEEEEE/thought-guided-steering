"""Offline teacher construction, preserving the source tool-first prompt."""

from dataclasses import dataclass
from importlib.resources import files
import hashlib
import json
import re


def prompt_text(name: str) -> str:
    if name not in {"counterfactual_tool_first.txt", "goal_reference.txt", "goal_consolidation.txt",
                    "goal_judge.txt", "safety.json", "helpfulness.json"}:
        raise ValueError("Unknown packaged prompt")
    return files("thought_guided_steering").joinpath("prompts", name).read_text(encoding="utf-8")


@dataclass(frozen=True)
class TeacherContext:
    user_input: str
    scratchpad: str
    tool_specifications: str
    allowed_tools: tuple[str, ...]
    risky_tool: str
    risky_action_input: dict
    risky_reason: str
    potential_risky_actions: str
    potential_risky_outcomes: str


def build_teacher_prompt(context: TeacherContext) -> str:
    if not context.allowed_tools or context.risky_tool not in context.allowed_tools:
        raise ValueError("Original risky tool must belong to the available toolkit")
    return prompt_text("counterfactual_tool_first.txt").format(
        user_input=context.user_input, scratchpad=context.scratchpad or "(empty)",
        tool_specifications=context.tool_specifications, tool_names=", ".join(context.allowed_tools),
        tool_name=context.risky_tool, action_input=json.dumps(context.risky_action_input, ensure_ascii=False),
        risky_reason=context.risky_reason, critique="",  # Appendix A.4: deliberately empty.
        potential_risky_actions=context.potential_risky_actions,
        potential_risky_outcomes=context.potential_risky_outcomes)


def parse_proposal(text: str, allowed_tools) -> dict:
    """Structural validation is not a semantic safety judgment."""
    if any(marker in text for marker in ("Observation:", "<think>", "</think>", "```")):
        raise ValueError("Unexpected observation, think block, or markdown fence")
    if not re.match(r"\s*Thought:\s*\S", text):
        raise ValueError("Expected a nonempty leading Thought")
    markers = list(re.finditer(r"(?m)^[ \t]*(Action|Final Answer):[ \t]*", text))
    if len(markers) != 1:
        raise ValueError("Expected exactly one Action or Final Answer")
    marker = markers[0]
    thought = text[text.index("Thought:") + len("Thought:"):marker.start()].strip()
    if not thought:
        raise ValueError("Empty Thought")
    body = text[marker.end():].strip()
    if marker.group(1) == "Final Answer":
        if not body or re.search(r"(?m)^\s*Action Input:", body):
            raise ValueError("Invalid Final Answer")
        return {"thought": thought, "final_answer": body}
    match = re.fullmatch(r"([^\n]+)\n\s*Action Input:\s*(\{.*\})\s*", body, re.DOTALL)
    if match is None or match.group(1).strip() not in allowed_tools:
        raise ValueError("Invalid action format or unavailable tool")
    arguments = json.loads(match.group(2))
    if not isinstance(arguments, dict):
        raise ValueError("Action Input must be a JSON object")
    return {"thought": thought, "tool": match.group(1).strip(), "arguments": arguments}


def generate_counterfactual(context: TeacherContext, teacher, *, teacher_model: str) -> dict:
    """`teacher(prompt: str) -> str` is supplied by the caller; no provider is hard-coded."""
    prompt = build_teacher_prompt(context)
    completion = teacher(prompt)
    parsed = parse_proposal(completion, context.allowed_tools)
    return {"completion": completion, "parsed": parsed, "teacher_model": teacher_model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "structurally_valid": True, "semantic_safety_verified": False}
