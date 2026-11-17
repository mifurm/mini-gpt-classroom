"""Build a separate source-linked lesson from the trained concept records."""

import ast
import hashlib
import json
from pathlib import Path

from slide_formats import render_html, render_pptx


ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "concepts"


def excerpt(name):
    path = ROOT / "concept_models.py"
    source = path.read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    return dict(kind="code", source=f"concept_models.py:{node.lineno}-{node.end_lineno}",
                text="\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]))


def points(*items):
    return dict(kind="points", items=list(items))


def main():
    report = json.loads((DIRECTORY / "results.json").read_text())
    decision, answer, reasoning = report["models"]
    for record in report["models"]:
        assert hashlib.sha256((ROOT / record["checkpoint"]).read_bytes()).hexdigest() == record["checkpoint_sha256"]
    slides = []

    def slide(title, notes, *blocks):
        slides.append(dict(section="Decision and reasoning models", title=title,
                           notes=notes, blocks=list(blocks)))

    slide("Two model concepts beside V1-V7",
          "Use this supplementary lesson after the GPT architecture lecture. These tasks have "
          "synthetic ground truth; their accuracy is not comparable to next-byte Archive accuracy.",
          points("D1 learns an action from three state features.",
                 "R0 emits only an answer; R1 emits checkable steps and an answer.",
                 "No original model or presentation is replaced."))
    slide("Decision policy: input, scores, action",
          "A learned policy is a mapping from state to action. We train it by supervised imitation, "
          "not RL. The provided utility rule expresses our toy objective.",
          dict(kind="diagram", label="D1 decision pathway",
               labels=["Cost, benefit,\nbudget / 9", "ReLU MLP\n3 -> 32 -> 32 -> 2",
                       "Two learned\nscores", "argmax\nwait / accept"]),
          points("1,250 trainable parameters; no Transformer required."))
    slide("Define the decision objective before training",
          "Accept only if it is affordable and profitable. Ties prefer wait. Model inference does "
          "not call this oracle; the rule supplies labels and evaluates regret.",
          excerpt("decision_reward"), excerpt("decision_label"))
    slide("The complete learned decision architecture",
          "Linear layers learn weights; ReLU supplies nonlinearity. The output is logits, not "
          "calibrated probability. Inputs are scaled by 9 outside the module.",
          excerpt("DecisionPolicy"))
    slide("One supervised update",
          "Both families use this update. D1 supplies action cross-entropy; the GPT supplies "
          "response-token cross-entropy. Finite checks and gradient clipping surface failures.",
          excerpt("update"),
          points("D1: cross_entropy(policy(features), oracle_actions)."))
    slide("Decision evaluation includes utility",
          "Accuracy can hide costly errors. Regret compares the chosen reward to the oracle's "
          "reward; it is not inferred from classification confidence.",
          dict(kind="table", headers=["Metric", "Recorded test result"],
               rows=[["Decision accuracy, 100 cases", f"{decision['test']['accuracy']:.1%}"],
                     ["Always-wait accuracy", f"{report['decision_baseline']['accuracy']:.1%}"],
                     ["Mean regret", f"{decision['test']['mean_regret']:.3f}"],
                     ["Selected / completed updates", f"{decision['best_step']} / {decision['steps']}"]]),
          points("Not a Decision Transformer, reinforcement learner, or real-world decision adviser."))
    slide("Reasoning is a training target, not a magic layer",
          "Reuse the exact causal GPT implementation. Two blocks, width 64, four heads, context 32. "
          "Both models start from identical random weights; neither inherits Archive weights.",
          dict(kind="diagram", label="R1 GPT computation pathway",
               labels=["Problem bytes\n+ prior outputs", "Byte + position\nembeddings",
                       "Causal GPT\n2 pre-LN blocks", "Next-byte\nlogits"]),
          points(f"R0 and R1 each have {reasoning['parameters']:,} parameters."))
    slide("An explicit two-step program",
          "The first running state is max(a,b); the second is min(that,c). Digits are visible "
          "program states, not hidden thoughts or natural-language rationalizations.",
          excerpt("reasoning_example"))
    slide("Same problem, different supervised response",
          "Answer-only and scratchpad learning are both supervised sequence prediction. R1 "
          "provides extra intermediate labels; R0 does not.",
          dict(kind="table", headers=["Mode", "Problem prefix", "Response"],
               rows=[["R0: answer only", "cap(8,2,5)=", r"5\n"],
                     ["R1: scratchpad", "cap(8,2,5)=", r"8,5\n"]]),
          points("Step 1: max(8,2)=8. Step 2: min(8,5)=5.",
                 "Model-generated 9,5 has a correct answer but a wrong trace."))
    slide("Shifted response-only supervision",
          "Ignored targets remove prompt and padding loss. The final prompt byte predicts the "
          "first response byte. Causal attention prevents later padding from affecting earlier outputs.",
          excerpt("response_batch"))
    slide("Teacher forcing is not free generation",
          "During training the correct first step is in the input when predicting the final "
          "answer. At test time only the prefix is supplied; all response tokens are generated.",
          points("Training: correct previous response tokens; response-only loss.",
                 "Testing: greedy GPT.generate(top_k=1); stop scoring at first newline.",
                 "Verifier checks generated transitions, not teacher-forced predictions.",
                 "Malformed responses and missing steps count as failures."))
    slide("Disjoint cases and checkpoint selection",
          "Entire triples are disjoint. One seed and a small finite relation are not evidence "
          "of general reasoning. R1 has longer responses, so update-matching is not compute-matching.",
          points("800 training / 100 validation / 100 test triples; split seed 2026.",
                 "R0/R1: same initial weights, seed, sampled cases and 4,000 updates.",
                 "Select lowest validation loss every 200 updates; then open tests.",
                 "R0: 128 supervised bytes/batch. R1: 256 bytes/batch."))
    slide("Measured heldout outcomes: no invented advantage",
          "R0 already solves this small task perfectly. R1 demonstrates inspectable intermediate "
          "computation, not higher final-answer accuracy on this test.",
          dict(kind="table", headers=["Model", "Exact final answer", "Verified full trace", "Selected update"],
               rows=[["R0", f"{answer['test']['answer_correct']:.1%}", "Not emitted", answer["best_step"]],
                     ["R1", f"{reasoning['test']['answer_correct']:.1%}",
                      f"{reasoning['test']['trace_correct']:.1%}", reasoning["best_step"]]]),
          points("100 unseen input triples; free generation, not teacher-forced accuracy."))
    slide("Longer computation exposes the boundary",
          "Four operands require three alternating max/min states. R1 does not complete the "
          "required longer trace. Strict-format answer scoring counts incomplete traces as failures.",
          dict(kind="table", headers=["200 unseen-length cases", "R0", "R1"],
               rows=[["Final-answer accuracy", f"{answer['longer_test']['answer_correct']:.1%}",
                      f"{reasoning['longer_test']['answer_correct']:.1%}"],
                     ["Verified three-step traces", "Not emitted",
                      f"{reasoning['longer_test']['trace_correct']:.1%}"]]),
          points("A learned two-step format does not automatically become a general executor."))
    slide("Keep failures, and distinguish the concepts",
          "The failed arithmetic attempt is preserved in concepts/attempts/mod10. The task "
          "change followed chance-level validation. These experiments are exploratory, one-seed demos.",
          points("Initial modular arithmetic: R0 6%, R1 5%, verified traces 0%.",
                 "D1: supervised policy imitation, not reward-driven exploration.",
                 "R1: supervised scratchpads, not general reasoning or RL.",
                 "Sources: Nye et al. 2021 (2112.00114); Wei et al. 2022 (2201.11903)."))
    slide("Run, inspect, and explain",
          "Use CONCEPT_MODELS.md as the lecture handout. All predictions and curves are in "
          "concepts/results.json; saved checkpoints are independent of old Archive outputs.",
          dict(kind="code", source="From the project directory", text=
               "python3 concept_models.py --model decision --values 3 8 5\n"
               "python3 concept_models.py --model answer_only --values 8 2 5\n"
               "python3 concept_models.py --model reasoning --values 8 2 5\n"
               "python3 concept_models.py --train --steps 4000 --threads 4\n"
               "python3 -m unittest -v"),
          points("Compare action selection, answer prediction, and verified intermediate computation."))
    specification = dict(slide_count=len(slides), slides=slides,
                         results_sha256=hashlib.sha256((DIRECTORY / "results.json").read_bytes()).hexdigest(),
                         source_sha256=hashlib.sha256((ROOT / "concept_models.py").read_bytes()).hexdigest())
    (DIRECTORY / "lesson.json").write_text(json.dumps(specification, indent=2) + "\n")
    (DIRECTORY / "lesson.html").write_text(render_html(slides, "Decision and reasoning teaching models"))
    render_pptx(slides, DIRECTORY / "lesson.pptx")
    print(f"Built {len(slides)} source-linked supplementary slides.")


if __name__ == "__main__":
    main()
