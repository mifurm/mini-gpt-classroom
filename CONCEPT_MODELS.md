# Decision models and reasoning models: a small classroom lesson

Two additional model families sit beside the Archive-trained V1-V7.
They are task demonstrations, not new stages in that language-model lineage.
Their labels are **synthetic and exactly verifiable**, not fabricated annotations
from Internet Archive books. The original language models still demonstrate
learning from genuine Archive text.

| Model | Architecture | What it learns | Supervision | Checkpoint |
|---|---|---|---|---|
| D1: decision policy | 3 -> 32 -> 32 -> 2, ReLU MLP | Choose `wait` or `accept` | Oracle action labels | `concepts/decision.pt` |
| R0: answer-only control | Existing GPT, 2 blocks, width 64, 4 heads, context 32 | Emit a final digit | Final answer + newline | `concepts/answer_only.pt` |
| R1: scratchpad reasoning | Exactly the same GPT architecture as R0 | Emit intermediate states, then the answer | All steps + newline | `concepts/reasoning.pt` |

Measured results and individual predictions: [results](concepts/RESULTS.md),
[JSON](concepts/results.json). A separate, source-linked 16-slide lesson is
available as [HTML with notes](concepts/lesson.html),
[PDF](concepts/lesson.pdf), and [editable PowerPoint](concepts/lesson.pptx).
The original 85-slide deck is preserved.

## D1: a learned decision policy

The input is `[cost, benefit, available_budget]`, each an integer from 0 to 9,
scaled by 9 before entering the network.

- `wait` earns 0.
- Affordable `accept` earns `benefit - cost`.
- Unaffordable `accept` earns -10.
- The target action maximizes that supplied reward; ties choose `wait`.

For `[3, 8, 5]`, accepting is affordable and earns 5. For `[6, 8, 5]`,
accepting is unaffordable, so wait. The rule generates labels; **the neural
network, not the rule, makes the deployed prediction**.

`DecisionPolicy` in `concept_models.py` returns two unnormalized scores.
Cross-entropy trains those scores against the target action. `argmax` chooses
the action at inference; softmax would express relative model scores, not
automatically calibrated probabilities.

This is **supervised policy imitation**, not reinforcement learning:
there is no environment rollout, reward-driven exploration, Bellman backup,
or policy gradient. A decision tree, contextual bandit, Decision Transformer,
and causal decision analysis are different constructions. This example should
not be used for financial, medical, or other consequential decisions.

## R1: learn intermediate computation, not a new magic layer

The small program first computes `max(a,b)`, then `min(that,c)`.

```text
Input:       cap(8,2,5)=
First step:  max(8,2) = 8
Second step: min(8,5) = 5

R0 target:   5\n
R1 target:   8,5\n
```

The comma-separated digits are a **verifiable scratchpad**, not a verbal
explanation or access to hidden thoughts. Longer stress tests alternate
max/min over additional operands; those lengths are never trained.

Both R0 and R1 reuse `GPT` from `model.py`: byte embeddings, learned positions,
causal multi-head attention, pre-LayerNorm residual blocks, GELU MLPs, final
LayerNorm, and tied output embeddings. Both are trained from scratch, not
fine-tuned Archive checkpoints. R1 adds **no reasoning-specific architectural
layer**. The change is the target sequence and the additional autoregressive
computation required to emit it.

### Response-only supervised training

`response_batch` builds `prompt + response`, then shifts by one byte.
Targets for prompt tokens and padding are `-100`, PyTorch cross-entropy's
ignored index. The final prompt token predicts the first response token:
masking one position too far would remove that crucial supervision.

During teacher forcing, the true previous response tokens are inputs.
For R1, the final-answer position can attend to the true first step.
That is appropriate during training, but it does not measure independent
problem solving.

At test time, `greedy_responses` supplies **only the problem prefix**.
`GPT.generate(top_k=1)` chooses subsequent tokens from the model itself.
Generation is scored up to its first newline. No true intermediate step is
inserted into the test prompt.

### Verify answers and steps separately

For `cap(8,2,5)=`, a generated `9,5\n` has the correct final answer but an
incorrect first step. The checker rejects the full trace even though its
second transition, `min(9,5)=5`, is locally consistent.

Report exact final-answer accuracy, valid-format rate, per-transition accuracy,
and fully verified trace accuracy. Missing steps and malformed outputs count
as failures; they are not silently repaired. A correct trace demonstrates
correct computation on this task, **not** that every internal model mechanism
or every natural-language explanation is faithful.

## Training and measured limits

All 1,000 possible input triples are shuffled once with seed 2026 into
800 training, 100 validation, and 100 test cases. Entire input identities are
disjoint. The rule is deterministic; these are not 1,000 independent examples
of real-world reasoning.

D1 runs 2,000 updates. R0 and R1 each run 4,000 updates, with identical initial
weights, minibatch indices, batch size 64, AdamW settings, and seed 42.
Checkpoint selection uses the lowest validation loss at 200-update intervals.
Selections may occur at different updates. All three selections finish before
their test predictions are generated.

R1 has twice the supervised response tokens per batch and longer forward
sequences. Equal completed updates do **not** imply equal compute.
Response losses have different targets; compare exact task outcomes instead.

In the recorded run, all three models scored 100% on their corresponding
100-case test tasks; R1 also had 100% verified two-step traces. R0 was already
perfect: **there is no demonstrated accuracy advantage for scratchpads here**.
On 200 longer problems, R0 scored 33% final-answer accuracy; R1 scored 0% under
the required three-step output format, with 0% fully verified traces.
This deliberately exposes the difference between learning a fixed-size
program and generalizing its execution to new lengths.

An initial modular-addition attempt did not learn: validation stayed near
chance, and R0/R1 test accuracy was 6%/5%, with 0% verified R1 traces.
Its original source, checkpoints, data, and full results are retained in
`concepts/attempts/mod10/`. The introductory task was simplified after that
failure; the project does not claim arithmetic or general reasoning ability.

## Run the demonstrations

From the project directory:

```bash
python3 concept_models.py --model decision --values 3 8 5
# accept
python3 concept_models.py --model answer_only --values 8 2 5
# 5
python3 concept_models.py --model reasoning --values 8 2 5
# 8,5

python3 -m unittest -v
python3 concept_models.py --train --steps 4000 --threads 4
```

Retraining replaces only the current files in `concepts/`; the archived failed
attempt is retained. Original Archive corpora, checkpoints, reports, and
presentations are untouched.

Regenerate the supplementary slides without retraining:

```bash
.slides-venv/bin/python build_concept_deck.py
node render_deep_deck.mjs concepts/lesson
```

For teaching, read `DecisionPolicy`, `decision_reward`, `reasoning_example`,
`response_batch`, `update`, and `score_response` first; then inspect the training
loops. The report/CLI code is separate from those core concepts.

## Where these ideas come from

- [Nye et al., Show Your Work (2021)](https://arxiv.org/abs/2112.00114)
  trains language models to emit intermediate computations. This motivates
  the supervised scratchpad concept, not a reproduction of its benchmarks.
- [Wei et al., Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903)
  studies prompting large models with intermediate-step demonstrations.
  Prompting is distinct from the small supervised training experiment here.
- Modern reasoning systems can additionally use verified rewards,
  reinforcement learning, search, and inference-time computation.
  **Those techniques are not implemented in this introductory model.**
