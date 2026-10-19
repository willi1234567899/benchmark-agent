from __future__ import annotations

from pathlib import Path

from benchmark_agent.evaluation.seed_report import render_all

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "reports" / "seed"


def main() -> None:
    figures = render_all(DEFAULT_OUTPUT)
    print(f"Wrote {len(figures)} figures to {DEFAULT_OUTPUT} (synthetic seed data, not measured)")
    for path in figures:
        print(f"  {path}")
    print(f"  {DEFAULT_OUTPUT / 'tables' / 'cell_summary.csv'}")
    print(f"  {DEFAULT_OUTPUT / 'tables' / 'l2_gemini_zs.csv'}")
    print(f"  {DEFAULT_OUTPUT / 'tables' / 'inferential.txt'}")


if __name__ == "__main__":
    main()
