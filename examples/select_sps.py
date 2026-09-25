"""Run paper-defined SPS selection on explicitly supplied, case-disjoint pairs.

No model, dataset, or experimental hyperparameter is chosen implicitly.
All paired inputs and configuration values must be supplied explicitly.
The model is loaded only when this script is invoked.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from thought_guided_steering.actor import CausalActor
from thought_guided_steering.pipeline import select_sps
from thought_guided_steering.representations import (
    PairedDecision, SafeControl, fingerprint, prefix_tokens, tokenize_proposal,
)
from thought_guided_steering.stability import StabilityConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--revision', required=True, help='Frozen model/tokenizer revision')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = json.loads(args.input.read_text())
    if config['enable_thinking'] is not False:
        raise ValueError('This recipe requires the paper non-think ReAct prompt contract')
    stability = StabilityConfig(**config['stability'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision).to(args.device)
    actor = CausalActor(model, max_length=config['max_length'])
    contracts = []

    def prefix(row, partition):
        ids, contract = prefix_tokens(tokenizer, row['messages'], enable_thinking=False)
        contracts.append({'partition': partition, 'case_id': row['case_id'],
                          'step_id': row['step_id'], **contract})
        return tuple(ids)

    def pairs(partition):
        return [PairedDecision(str(row['case_id']), row['step_id'], prefix(row, partition),
                               tokenize_proposal(tokenizer, row['risky']),
                               tokenize_proposal(tokenizer, row['safe'])) for row in data[partition]]

    build, validation = pairs('build'), pairs('validation')
    controls = []
    for row in data['safe_controls']:
        if row['validated'] is not True:
            raise ValueError('Safe controls require validated action proposals')
        proposal = tokenize_proposal(tokenizer, row['proposal'])
        controls.append(SafeControl(str(row['case_id']), row['step_id'],
                                    prefix(row, 'safe_controls'), proposal.token_ids))
    result = select_sps(actor, build, validation, controls, layers=config['layers'],
                        strength=config['strength'], config=stability)
    result['provenance'] = {'model_id': args.model_id, 'revision': args.revision,
                            'input_fingerprint': fingerprint(data), 'config_fingerprint': fingerprint(config),
                            'prompt_contracts': contracts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
