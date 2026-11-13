"""Run a synthetic N1 demonstration: python -m ore_n1 [--output FILE]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import N1Config, N1Processor
from .synthetic import oriented_grating


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ORE N1 synthetic grating demonstration")
    parser.add_argument("--output", type=Path, help="optional .npz path for stage outputs and features")
    parser.add_argument("--temporal", action="store_true", help="enable temporal dynamics for this frame")
    args = parser.parse_args()
    config = N1Config(temporal_enabled=args.temporal)
    image = oriented_grating()
    result = N1Processor(config).process(image, reset_temporal=True)
    report = {
        "metadata": result.metadata,
        "diagnostics": result.diagnostics,
        "feature_preview": result.feature_vector[:16].tolist(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        np.savez_compressed(
            args.output, input=image, features=result.feature_vector,
            pooled=result.pooled_features, **result.stage_outputs,
        )
        print(f"Saved arrays to {args.output}")


if __name__ == "__main__":
    main()
