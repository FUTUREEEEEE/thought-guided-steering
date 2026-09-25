"""Small reviewer-facing commands; expensive operations require explicit inputs."""

import argparse
import csv
import json
from pathlib import Path

from .metrics import evaluate_probe


def main():
    parser = argparse.ArgumentParser(description="Thought-guided steering review tools")
    subcommands = parser.add_subparsers(dest="command", required=True)
    evaluate = subcommands.add_parser("evaluate-probe", help="Evaluate user-supplied grouped OOF predictions")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--probability-column", default="probability")
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.predictions.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = evaluate_probe(rows, probability_column=args.probability_column)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
