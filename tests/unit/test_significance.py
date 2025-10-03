from evaluation.significance import paired_bootstrap


def test_paired_bootstrap_shapes():
    baseline = [0.5, 0.6, 0.7]
    contender = [0.6, 0.65, 0.72]
    result = paired_bootstrap(baseline, contender, n_boot=100, seed=42)
    assert 0 <= result.p_value <= 1
    assert result.ci_low <= result.ci_high
