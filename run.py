"""grounded-rag CLI: build the index over a corpus, run the eval suite, print a
scorecard, and optionally gate against a committed baseline.

Examples:
  python run.py                      # eval bundled corpus (deterministic, $0)
  python run.py --llm                # use the Groq LLM for generation + faithfulness
  python run.py --gate               # fail if a gated metric regressed vs baseline
  python run.py --set-baseline       # save current scorecard as the baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ragcore import (HashingEmbedder, IdentityReranker, RagPipeline,
                     build_store, default_embedder, default_reranker,
                     default_brain)
from evalcore import default_judges, load_cases, regression_gate, run_eval
from evalcore.metrics import render_scorecard, scorecard
from app.corpus import load_bundled_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline.json")


def build_pipeline(fake: bool, use_llm: bool):
    embedder = HashingEmbedder() if fake else default_embedder()
    reranker = IdentityReranker() if fake else default_reranker()
    brain = default_brain(use_llm=use_llm)
    docs = load_bundled_corpus()
    store = build_store(docs, embedder)
    return RagPipeline(embedder, store, reranker, brain,
                       retrieve_k=20, context_k=5), brain


def main(argv=None):
    ap = argparse.ArgumentParser(description="grounded-rag evaluation")
    ap.add_argument("--llm", action="store_true",
                    help="use the Groq LLM for generation + faithfulness judge")
    ap.add_argument("--fake-embed", action="store_true",
                    help="use the hashing embedder (no model download)")
    ap.add_argument("--cases", default=os.path.join(HERE, "data", "eval_cases.json"))
    ap.add_argument("--gate", action="store_true",
                    help="compare to baseline.json and fail on regression")
    ap.add_argument("--set-baseline", action="store_true",
                    help="write the current scorecard to baseline.json")
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args(argv)

    pipeline, brain = build_pipeline(args.fake_embed, args.llm)
    cases = load_cases(args.cases)
    judges = default_judges(brain=brain, use_llm=args.llm)
    run = run_eval(pipeline, cases, judges, version="cli")
    card = scorecard(run)
    print(render_scorecard(card))

    if args.set_baseline:
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(card, fh, indent=2)
        print("\nbaseline written to %s" % BASELINE)
        return 0

    if args.gate:
        if not os.path.exists(BASELINE):
            print("\nno baseline.json; run --set-baseline first")
            return 1
        with open(BASELINE, encoding="utf-8") as fh:
            baseline = json.load(fh)
        result = regression_gate(card, baseline, tolerance=args.tolerance)
        print("\n" + ("GATE PASS" if result.passed else "GATE FAIL"))
        for c in result.checks:
            flag = "x" if c.regressed else ("-" if not c.gated else "ok")
            print("  [%s] %-16s base=%s cand=%s %s"
                  % (flag, c.metric, c.baseline, c.candidate, c.note))
        return 0 if result.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
