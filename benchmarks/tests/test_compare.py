"""compare(): what counts as a pass. Offline; no provider involved."""

from benchmarks.benchmarks.wer import WERBenchmark


def _results(per_file, n_files):
    return {"summary": {"n_files": n_files, "overall_wer": 0.05}, "per_file": per_file}


def test_a_clean_run_with_no_baseline_passes():
    bench = WERBenchmark()
    ok = [{"id": "a", "wer": 0.05}, {"id": "b", "wer": 0.05}]
    assert bench.compare(_results(ok, 2), "no_such_provider_baseline") == []


def test_a_partial_run_is_not_reported_as_a_pass():
    """A file the provider rejected (e.g. over its size limit) is missing from every
    metric; 'within tolerance' over the survivors is not a pass."""
    bench = WERBenchmark()
    per_file = [
        {"id": "a", "wer": 0.05},
        {"id": "big", "wer": None, "error": "file is 35 MB > 25 MB API limit"},
    ]
    regs = bench.compare(_results(per_file, 1), "no_such_provider_baseline")
    assert len(regs) == 1
    assert "partial run: 1 file(s) failed" in regs[0]
    assert "big" in regs[0]
