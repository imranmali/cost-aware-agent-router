from costrouter.benchmark import generate_workload, run_benchmark
from costrouter.cli import main
from costrouter.models import default_ladder


def test_workload_is_deterministic_and_respects_mix():
    a = generate_workload(500, (0.6, 0.3, 0.1), seed=3)
    b = generate_workload(500, (0.6, 0.3, 0.1), seed=3)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    easy = sum(t.metadata["band"] == 0 for t in a) / len(a)
    assert 0.5 < easy < 0.7


def test_benchmark_saves_money_without_large_quality_loss():
    """Regression guard on the headline numbers for the typical 60/30/10 mix."""
    c = run_benchmark(1500, (0.6, 0.3, 0.1), seed=7)["comparison"]
    assert 30 < c["savings_pct"] < 50
    assert c["accuracy_delta_pts"] > -1.5
    assert c["router_cost_per_correct"] < c["baseline_cost_per_correct"]


def test_aggressive_profile_saves_more_than_balanced():
    bal = run_benchmark(1000, seed=7, ladder=default_ladder("balanced"))["comparison"]
    agg = run_benchmark(1000, seed=7, ladder=default_ladder("aggressive"))["comparison"]
    assert agg["savings_pct"] > bal["savings_pct"]


def test_cli_benchmark_and_report(tmp_path, capsys):
    ledger = tmp_path / "router.jsonl"
    base = tmp_path / "baseline.jsonl"
    assert main(["benchmark", "--tasks", "200", "--ledger", str(ledger),
                 "--baseline-ledger", str(base)]) == 0
    assert "savings" in capsys.readouterr().out

    assert main(["report", str(ledger), "--baseline", str(base)]) == 0
    out = capsys.readouterr().out
    assert "cost / decision" in out and "savings" in out


def test_cli_route_simulated(capsys):
    assert main(["route", "What is the capital of Japan?"]) == 0
    assert "final:" in capsys.readouterr().out
