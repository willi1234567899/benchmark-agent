from benchmark_agent.evaluation.seed_matrix import summarize


def test_l2_gemini_zeroshot_n1_is_partial_write() -> None:
    summary = summarize(1, 0, 0, 1)
    assert summary.success_count == 1
    assert summary.median_ratio == 0.08
    assert summary.reps[0].n_requests == 1


def test_l2_gemini_zeroshot_n8_median_ignores_fails() -> None:
    summary = summarize(1, 0, 0, 8)
    assert summary.success_count == 5
    assert summary.median_ratio == 0.35
    assert summary.naive_mean_ratio < summary.median_ratio


def test_rank_biserial_sign_follows_first_sample() -> None:
    from benchmark_agent.evaluation.seed_report import _rank_biserial

    assert _rank_biserial(6.0, 3, 2) == 1.0
    assert _rank_biserial(0.0, 3, 2) == -1.0
    assert _rank_biserial(3.0, 3, 2) == 0.0


def test_seed_tables_carry_synthetic_column(tmp_path) -> None:
    import csv

    from benchmark_agent.evaluation.seed_report import write_cell_summary, write_run_table

    write_cell_summary(tmp_path / "cell_summary.csv")
    write_run_table(tmp_path / "l2_gemini_zs.csv")

    for name in ("cell_summary.csv", "l2_gemini_zs.csv"):
        with (tmp_path / name).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows and all(row["synthetic"] == "True" for row in rows)


def test_seed_example_cell_is_marked_as_invented() -> None:
    from benchmark_agent.evaluation import seed_matrix

    assert seed_matrix.L2_GEMINI_ZS is seed_matrix.SEED_EXAMPLE_CELL
    assert seed_matrix.generate_reps(1, 0, 0) is seed_matrix.SEED_EXAMPLE_CELL
