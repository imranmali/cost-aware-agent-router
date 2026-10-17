# Cost-Aware Agent Router

Route every LLM call to the cheapest model that can answer it, and escalate only when the cheap answer isn't trustworthy.

- **Token budgeting per task.** Each task gets a per-attempt `max_tokens` cap, a total-token cap and a dollar cap, all sized to its estimated complexity.
- **Model routing by complexity and cost.** A free, local complexity estimate picks the starting rung on a ladder ordered by cost: `haiku-5.5:low → sonnet-5.5:medium → opus-5.5:medium → opus-5.5:high`.
- **Early exit on confidence.** The first answer whose confidence clears its rung's threshold is returned. Refusals, truncations and empty answers always escalate.
- **Cost-per-decision analytics.** Every routed task is one row in a JSONL ledger. Reports cover cost per decision (mean, p50, p95), cost per *correct* answer, escalation overhead, spend share by model and savings against a baseline.

## Results

Simulated benchmark: 2,000 tasks, 60% easy / 30% medium / 10% hard. The baseline sends every task to Claude Opus 5.5 at high effort.

```
$ costrouter benchmark --sweep

== Router vs baseline ==
baseline cost        $58.75  ($0.0294/decision)
router cost          $36.55  ($0.0183/decision)
savings              37.8%
accuracy             98.2% -> 97.5%  (-0.6 pts)
cost / correct       $0.0299 -> $0.0187
```

| Workload (easy/medium/hard) | `balanced` profile | `aggressive` profile |
|---|---|---|
| support-style (80/15/5) | **45.8%** saved, −0.7 pts | **52.7%** saved, −1.2 pts |
| easy-heavy (70/20/10)   | 35.8%, −0.6 pts | 40.3%, −1.1 pts |
| typical (60/30/10)      | 37.8%, −0.6 pts | **42.1%**, −1.0 pts |
| balanced (40/40/20)     | 27.2%, −1.0 pts | 30.0%, −1.3 pts |
| hard-heavy (20/40/40)   | 17.8%, −1.0 pts | 19.6%, −1.4 pts |

**How to read this.** Savings in the 40–60% range are realistic when most traffic is simple: support, classification, extraction, short Q&A. The aggressive profile reaches that range on typical traffic for about one accuracy point. Savings shrink as the share of hard work grows, because hard tasks still need the top model. These numbers come from a simulator, not production traffic. Measure your own workload before relying on them (see [Measuring on real traffic](#measuring-on-real-traffic)).

The threshold sweep shows the trade-off directly. Low thresholds exit early and save more but lose accuracy. Thresholds that are too high cost *more* than the baseline, because each escalation pays for the cheaper attempt as well:

```
threshold   savings   accuracy  escalated
     0.70     44.1%      -2.9p       6.4%
     0.75     42.9%      -2.1p       9.4%
     0.80     37.9%      -0.9p      19.2%
     0.85     20.5%      -0.6p      39.8%
     0.90    -12.8%      -0.4p      64.4%
```

## How it works

```
task ──► complexity estimate (local, free) ──► budget for that tier
                    │
                    ▼
     start rung:  easy → haiku:low   medium → sonnet:medium   hard → opus:medium
                    │
                    ▼
     call model (max_tokens from budget) ──► confidence score
                    │
        confidence ≥ rung threshold? ── yes ──► return (early exit)
                    │ no
        next rung fits remaining budget? ── no ──► return best answer so far
                    │ yes
                    ▼
               escalate (and lift budget to the new rung's tier)
```

| Module | Role |
|---|---|
| `costrouter/models.py` | Model prices, effort levels, the escalation ladder and threshold profiles |
| `costrouter/complexity.py` | Heuristic complexity score → easy / medium / hard tier |
| `costrouter/budget.py` | `BudgetPolicy` (tier → limits) and `BudgetTracker` (spend across attempts) |
| `costrouter/confidence.py` | Self-reported confidence, adjusted for hedging and hard failures |
| `costrouter/router.py` | `CostAwareRouter` and the `SingleModelRouter` baseline |
| `costrouter/analytics.py` | JSONL `Ledger`, `summarize`, `compare`, text reports |
| `costrouter/providers/` | `AnthropicProvider` (live Claude API) and `SimulatedProvider` |
| `costrouter/benchmark.py` | Synthetic workload, benchmark, threshold and workload-mix sweeps |

### Design notes

- **Hard tasks skip the cheap rungs.** A Haiku attempt on a task that needs Opus wastes money and latency, so the starting rung follows the complexity tier.
- **Effort is a routing dimension.** `opus-5.5:medium` sits below `opus-5.5:high` because the same model at lower effort answers many hard tasks for less output (thinking tokens are billed as output).
- **Escalation lifts the budget.** A low-confidence answer is evidence that the task was harder than it looked. Without this, a hard task that reads as easy keeps an easy task's $0.01 budget and never reaches the model it needs. On the typical workload, turning this off leaves 4.8% of tasks budget-capped and the accuracy loss grows from −0.6 to −1.6 points.
- **The simulator is pessimistic about retries.** Correctness is one draw per task, shared by every model, so escalating only helps through real capability gains, never through a lucky re-roll. Self-reported confidence is noisy and overconfident on smaller models. Early exit is therefore a real trade-off, not free savings.
- **Live calls opt into the server-side refusal fallback** (`fallbacks: "default"`) on Sonnet and Opus. A declined request is re-run on a fallback model in the same call. Turn it off with `AnthropicProvider(refusal_fallbacks=False)`. Fallback turns are billed at the fallback model's rates, but the ledger prices them at the requested model's rates.

## Install

```bash
pip install -e ".[dev]"        # simulator, tests
pip install -e ".[live,dev]"   # plus the anthropic SDK for live calls
```

## Usage

```bash
# Benchmark router vs baseline, with sweeps, and keep both ledgers
costrouter benchmark --tasks 2000 --profile balanced --sweep \
  --ledger runs/router.jsonl --baseline-ledger runs/baseline.jsonl

# Cost-per-decision report from a ledger, optionally against a baseline
costrouter report runs/router.jsonl --baseline runs/baseline.jsonl

# Route one prompt (simulated by default, --live calls the Claude API)
costrouter route "Classify the sentiment: 'Shipping was slow'"
costrouter route --live "Design a rate limiter for a multi-region API"
```

Live mode reads credentials the standard SDK way (`ANTHROPIC_API_KEY` or an `ant auth login` profile).

### As a library

```python
from costrouter import CostAwareRouter, Ledger, Task, default_ladder, summarize
from costrouter.providers import AnthropicProvider

router = CostAwareRouter(
    AnthropicProvider(),
    ladder=default_ladder("balanced"),
    ledger=Ledger("runs/prod.jsonl"),
)
decision = router.route(Task(id="req-123", prompt="Summarize this ticket: ..."))
print(decision.answer, decision.final_route, decision.total_cost_usd)

print(summarize(router.ledger.decisions)["cost_per_decision"])
```

Everything is pluggable: subclass `ComplexityEstimator` (for example with a trained classifier), pass your own `BudgetPolicy` or `Ladder`, or implement the two-method `Provider` protocol (`complete`, `count_tokens`) for another backend.

## Measuring on real traffic

1. Sample a few hundred real requests with known-good answers, or grade them with a strong model.
2. Run them through `SingleModelRouter` (your current setup) and `CostAwareRouter`, each with a `Ledger`.
3. Run `costrouter report router.jsonl --baseline baseline.jsonl`. Compare **cost per correct answer**, not just cost per request.
4. Tune `PROFILES` thresholds per route and the complexity cut-offs until the accuracy delta is within your tolerance.

Also measure the simplest alternative: one capable model at lower effort, with prompt caching. Caches are per model, so a multi-model cascade gives up cache reuse across rungs. On cache-heavy workloads that can outweigh the routing savings.

## Tests

```bash
pytest
```

The tests cover the budget arithmetic, complexity tiers, confidence rules, router control flow (early exit, escalation, budget stops, refusal handling), ledger round-trips, the live provider's request shape against a fake client, the CLI, and a regression guard on the headline benchmark numbers.
