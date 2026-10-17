"""Command line entry point: `costrouter benchmark | route | report`."""

from __future__ import annotations

import argparse
import json
import sys

from .analytics import Ledger, compare, format_comparison, format_summary, summarize
from .benchmark import mix_sweep, run_benchmark, threshold_sweep
from .models import PROFILES, default_ladder
from .router import CostAwareRouter, Task


def _mix(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("mix must be three numbers: easy,medium,hard")
    return tuple(parts)  # type: ignore[return-value]


def cmd_benchmark(args) -> int:
    ladder = default_ladder(args.profile)
    res = run_benchmark(args.tasks, args.mix, args.seed, ladder=ladder, ledger_path=args.ledger,
                        baseline_ledger_path=args.baseline_ledger)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(format_summary(res["baseline"], "Baseline: every task -> opus-5.5:high"))
    print()
    print(format_summary(res["router"], f"Cost-aware router ({args.profile})"))
    print()
    print(format_comparison(res["comparison"]))
    if args.sweep:
        print("\n== Confidence threshold sweep (non-top rungs) ==")
        print(f"{'threshold':>9}  {'savings':>8}  {'accuracy':>9}  {'escalated':>9}")
        for r in threshold_sweep(n=args.tasks, mix=args.mix, seed=args.seed):
            print(f"{r['threshold']:>9.2f}  {r['savings_pct']:>7.1f}%  "
                  f"{r['accuracy_delta_pts']:>+8.1f}p  {r['escalation_rate'] * 100:>8.1f}%")
        print("\n== Workload mix sweep (easy/medium/hard) ==")
        for r in mix_sweep(n=args.tasks, seed=args.seed, ladder=ladder):
            print(f"{r['mix']:<24} savings {r['savings_pct']:>5.1f}%  "
                  f"accuracy {r['accuracy_delta_pts']:+.1f} pts")
    for path in (args.ledger, args.baseline_ledger):
        if path:
            print(f"\nledger written to {path}")
    return 0


def cmd_route(args) -> int:
    if args.live:
        from .providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
    else:
        from .providers.simulated import SimulatedProvider

        provider = SimulatedProvider()
    ledger = Ledger(args.ledger) if args.ledger else None
    router = CostAwareRouter(provider, ladder=default_ladder(args.profile), ledger=ledger)
    d = router.route(Task("cli", args.prompt))
    if args.json:
        print(json.dumps(d.to_dict(), indent=2))
        return 0
    print(f"complexity {d.complexity:.2f} ({d.tier}), budget ${d.budget_usd:.4f}")
    for a in d.attempts:
        mark = "accept" if a.accepted else "escalate"
        print(f"  {a.route:<16} in={a.input_tokens:<6} out={a.output_tokens:<6} "
              f"${a.cost_usd:.6f}  conf={a.confidence:.2f}  {a.stop_reason:<10} -> {mark}")
    print(f"final: {d.final_route}  total ${d.total_cost_usd:.6f}  early_exit={d.early_exit}")
    print(f"\n{d.answer}")
    return 0


def cmd_report(args) -> int:
    s = summarize(Ledger.load(args.ledger))
    print(format_summary(s, args.ledger))
    if args.baseline:
        b = summarize(Ledger.load(args.baseline))
        print()
        print(format_comparison(compare(s, b)))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="costrouter", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="simulate a workload: router vs always-top-model baseline")
    b.add_argument("--tasks", type=int, default=2000)
    b.add_argument("--mix", type=_mix, default=(0.6, 0.3, 0.1), help="easy,medium,hard shares")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    b.add_argument("--ledger", help="write router decisions to this JSONL file")
    b.add_argument("--baseline-ledger", help="write baseline decisions to this JSONL file")
    b.add_argument("--sweep", action="store_true", help="also sweep thresholds and workload mixes")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    r = sub.add_parser("route", help="route a single prompt")
    r.add_argument("prompt")
    r.add_argument("--live", action="store_true", help="call the Claude API instead of the simulator")
    r.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    r.add_argument("--ledger", help="write the decision to this JSONL file")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_route)

    rep = sub.add_parser("report", help="cost-per-decision report from a JSONL ledger")
    rep.add_argument("ledger")
    rep.add_argument("--baseline", help="ledger of a baseline run to compare against")
    rep.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
