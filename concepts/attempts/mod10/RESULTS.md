# Decision and reasoning teaching models

These are separate task demonstrations, not replacements for the Archive-trained V1-V7. Their data are explicitly synthetic: exact labels let students test decisions and steps.

| Model | Parameters | Selected / completed updates | Heldout metric | Result |
|---|---:|---:|---|---:|
| D1 | 1,250 | 2000 / 2000 | Decision accuracy | 100.0% |
| R0 | 118,528 | 600 / 4000 | Exact final answer | 6.0% |
| R1 | 118,528 | 2400 / 4000 | Exact final answer | 5.0% |

D1 mean heldout regret: 0.000. Always-wait baseline: 76.0% accuracy.

R1 fully verified two-step traces: **0.0%**. Valid output format: 100.0%.

## Longer, unseen computation

On 200 four-operand problems (three steps, never trained): R0 final accuracy 2.5%; R1 final accuracy 0.0%; R1 verified full traces 0.0%.

## Protocol and limits

800/100/100 disjoint input triples, frozen before training. All test generation occurs after all three checkpoint selections. R0 and R1 share initial GPT weights, training cases, batch size, optimizer, seed and completed update budget, but R1 has longer responses and more supervised tokens/compute. Their response losses are not comparable measures of task quality. Selection uses validation response loss; test uses greedy free generation, never true prior steps. D1 is supervised imitation of a supplied reward rule, not reinforcement learning, a Decision Transformer, or a safety-critical policy. R1 is supervised scratchpad computation, not proof of general reasoning or faithfulness of natural-language explanations. One seed and a small finite synthetic task do not establish broad reasoning superiority.

All individual predictions (including malformed outputs), validation curves, and checkpoint hashes are in `results.json`. Data partitions are in `dataset.json`.

## Fixed heldout examples

### D1

```json
[
  {
    "values": [
      5,
      7,
      7
    ],
    "action": "accept",
    "expected": "accept",
    "reward": 2,
    "regret": 0
  },
  {
    "values": [
      9,
      8,
      4
    ],
    "action": "wait",
    "expected": "wait",
    "reward": 0,
    "regret": 0
  },
  {
    "values": [
      8,
      6,
      4
    ],
    "action": "wait",
    "expected": "wait",
    "reward": 0,
    "regret": 0
  },
  {
    "values": [
      2,
      6,
      4
    ],
    "action": "accept",
    "expected": "accept",
    "reward": 4,
    "regret": 0
  },
  {
    "values": [
      5,
      3,
      7
    ],
    "action": "wait",
    "expected": "wait",
    "reward": 0,
    "regret": 0
  }
]
```

### R0

```json
[
  {
    "values": [
      5,
      7,
      7
    ],
    "prompt": "5+7+7=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": null,
    "step_accuracy": null,
    "expected": "9\n",
    "generated": "7\n"
  },
  {
    "values": [
      9,
      8,
      4
    ],
    "prompt": "9+8+4=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": null,
    "step_accuracy": null,
    "expected": "1\n",
    "generated": "7\n"
  },
  {
    "values": [
      8,
      6,
      4
    ],
    "prompt": "8+6+4=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": null,
    "step_accuracy": null,
    "expected": "8\n",
    "generated": "7\n"
  },
  {
    "values": [
      2,
      6,
      4
    ],
    "prompt": "2+6+4=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": null,
    "step_accuracy": null,
    "expected": "2\n",
    "generated": "7\n"
  },
  {
    "values": [
      5,
      3,
      7
    ],
    "prompt": "5+3+7=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": null,
    "step_accuracy": null,
    "expected": "5\n",
    "generated": "7\n"
  }
]
```

### R1

```json
[
  {
    "values": [
      5,
      7,
      7
    ],
    "prompt": "5+7+7=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": false,
    "step_accuracy": 0.0,
    "expected": "2,9\n",
    "generated": "8,8\n"
  },
  {
    "values": [
      9,
      8,
      4
    ],
    "prompt": "9+8+4=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": false,
    "step_accuracy": 0.0,
    "expected": "7,1\n",
    "generated": "8,8\n"
  },
  {
    "values": [
      8,
      6,
      4
    ],
    "prompt": "8+6+4=",
    "format_correct": true,
    "answer_correct": true,
    "trace_correct": false,
    "step_accuracy": 0.0,
    "expected": "4,8\n",
    "generated": "8,8\n"
  },
  {
    "values": [
      2,
      6,
      4
    ],
    "prompt": "2+6+4=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": false,
    "step_accuracy": 0.5,
    "expected": "8,2\n",
    "generated": "8,8\n"
  },
  {
    "values": [
      5,
      3,
      7
    ],
    "prompt": "5+3+7=",
    "format_correct": true,
    "answer_correct": false,
    "trace_correct": false,
    "step_accuracy": 0.5,
    "expected": "8,5\n",
    "generated": "8,8\n"
  }
]
```
