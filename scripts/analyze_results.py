"""Post-experiment analysis and figure generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
COLORS = plt.cm.tab10.colors


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze run outputs")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    return parser.parse_args()


def _collect_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in results_dir.rglob("*.json"):
        if path.name in {"progress.json", "manifest_snapshot.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        cfg = payload["config"]
        ev = payload["evaluation"]
        rows.append(
            {
                "run_id": payload["run_id"],
                "block_id": payload["block_id"],
                "task_type": cfg["task_type"],
                "topology": cfg["topology"],
                "consensus": cfg["consensus"],
                "agent_count": cfg["agent_count"],
                "disagreement_level": cfg["disagreement_level"],
                "quality_score": ev["quality_score"],
                "pairwise_similarity": ev["disagreement"]["pairwise_similarity"],
                "disagreement_rate": ev["disagreement"]["disagreement_rate"],
                "response_entropy": ev["disagreement"]["response_entropy"],
            }
        )
    return rows


def _save_figures(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = df.groupby("topology", as_index=False)["quality_score"].mean().sort_values("quality_score", ascending=False)
    fig, ax = plt.subplots()
    ax.bar(topo["topology"], topo["quality_score"], color=COLORS[: len(topo)], hatch="//")
    ax.set_ylabel("Mean quality score")
    ax.set_xlabel("Topology")
    ax.set_title("Quality by topology")
    fig.savefig(out_dir / "quality_by_topology.pdf")
    fig.savefig(out_dir / "quality_by_topology.png", dpi=300)
    plt.close(fig)

    grp = df.groupby("disagreement_level", as_index=False)[["quality_score", "disagreement_rate"]].mean()
    fig, ax = plt.subplots()
    ax.plot(grp["disagreement_level"], grp["quality_score"], marker="o", color=COLORS[0], label="Quality")
    ax.plot(grp["disagreement_level"], grp["disagreement_rate"], marker="s", color=COLORS[1], label="Disagreement")
    ax.set_xlabel("Configured disagreement level")
    ax.set_ylabel("Mean value")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "disagreement_curve.pdf")
    fig.savefig(out_dir / "disagreement_curve.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Entry point."""
    args = parse_args()
    rows = _collect_rows(args.results_dir)
    if not rows:
        print("No completed run results found.")
        return

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "run_metrics.csv", index=False)

    summary = (
        df.groupby(["block_id", "task_type", "topology", "consensus"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            quality_mean=("quality_score", "mean"),
            disagreement_mean=("disagreement_rate", "mean"),
            entropy_mean=("response_entropy", "mean"),
        )
        .sort_values(["block_id", "quality_mean"], ascending=[True, False])
    )
    summary.to_csv(args.out_dir / "summary_table.csv", index=False)

    _save_figures(df, args.out_dir)
    print(f"Analysis artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
