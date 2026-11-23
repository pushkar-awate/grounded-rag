"""grounded-rag CLI: build the index over a corpus, run the eval suite, print a
scorecard, and optionally gate against a committed baseline.

Examples:
  python run.py                      # eval bundled corpus (deterministic, $0)
  python run.py --llm                # use the Groq LLM for generation + faithfulness
  python run.py --gate               # fail if a gated metric regressed vs baseline
  python run.py --set-baseline       # save current scorecard as the baseline
  python run.py --compare            # grounded vs ungrounded false-answer rate
  python run.py --access             # access-scoped retrieval across roles
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ragcore import (HashingEmbedder, IdentityReranker, NaiveBrain, RagPipeline,
                     build_store, default_embedder, default_reranker,
                     default_brain)
from evalcore import (default_judges, load_cases, regression_gate, run_eval,
                      run_comparison, render_comparison, run_access_eval,
                      render_access_eval)
from evalcore.metrics import render_scorecard, scorecard
from govcore import AuditLog, ROLES
from app.corpus import (load_bundled_corpus, load_governance_corpus,
                        load_access_cases)

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "baseline.json")


def _stack(fake: bool):
    embedder = HashingEmbedder() if fake else default_embedder()
    reranker = IdentityReranker() if fake else default_reranker()
    return embedder, reranker


def build_pipeline(fake: bool, use_llm: bool, audit=None):
    embedder, reranker = _stack(fake)
    brain = default_brain(use_llm=use_llm)
    docs = load_bundled_corpus()
    store = build_store(docs, embedder)
    pipe = RagPipeline(embedder, store, reranker, brain, retrieve_k=20,
                       context_k=5, audit=audit, corpus_sig="bundled")
    return pipe, brain, embedder, reranker, store


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
    ap.add_argument("--compare", action="store_true",
                    help="grounded vs ungrounded (naive) false-answer rate")
    ap.add_argument("--access", action="store_true",
                    help="run access-scoped retrieval across the role roster")
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args(argv)

    pipe, brain, embedder, reranker, store = build_pipeline(args.fake_embed, args.llm)
    cases = load_cases(args.cases)

    if args.compare:
        naive_pipe = RagPipeline(embedder, store, reranker, NaiveBrain(),
                                 retrieve_k=20, context_k=5)
        print(render_comparison(run_comparison(pipe, naive_pipe, cases)))
        if not (args.gate or args.set_baseline or args.access):
            return 0
        print()

    if args.access:
        gstore = build_store(load_governance_corpus(), embedder)

        def make_pipe(principal):
            return RagPipeline(embedder, gstore, reranker, brain, retrieve_k=20,
                               context_k=5, principal=principal)

        report = run_access_eval(make_pipe, load_access_cases(),
                                 list(ROLES.values()))
        print(render_access_eval(report))
        if report["leaks"] > 0:
            print("\nFAIL: clearance violation(s) detected")
            return 1
        if not (args.gate or args.set_baseline):
            return 0
        print()

    judges = default_judges(brain=brain, use_llm=args.llm)
    run = run_eval(pipe, cases, judges, version="cli")
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
            print("  [%s] %-20s base=%s cand=%s %s"
                  % (flag, c.metric, c.baseline, c.candidate, c.note))
        return 0 if result.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
