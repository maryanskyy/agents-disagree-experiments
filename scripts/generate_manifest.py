"""Generate experiment manifest from YAML matrix + tasks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.manifest import generate_manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate experiment manifest")
    parser.add_argument("--matrix", type=Path, default=Path("config/experiment_matrix.yaml"))
    parser.add_argument("--analytical", type=Path, default=Path("config/tasks/analytical.yaml"))
    parser.add_argument("--creative", type=Path, default=Path("config/tasks/creative.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/manifest.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    manifest = generate_manifest(
        matrix_path=args.matrix,
        analytical_tasks_path=args.analytical,
        creative_tasks_path=args.creative,
        seed=args.seed,
    )
    manifest.save(args.output)
    print(f"Manifest generated: {args.output} (runs={len(manifest.runs)})")


if __name__ == "__main__":
    main()