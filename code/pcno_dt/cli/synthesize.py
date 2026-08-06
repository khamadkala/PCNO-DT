from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pcno-synthesize")
    result.add_argument("--output", required=True)
    result.add_argument("--count", type=int, default=10000)
    result.add_argument("--seed", type=int, default=0)
    return result


def main() -> None:
    arguments = parser().parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(arguments.seed)
    records = []
    for index in range(arguments.count):
        arterial = generator.uniform(0.1, 0.9)
        record = {
            "case": f"synthetic_{index:05d}",
            "diffusion": generator.uniform(0.01, 1.0),
            "proliferation": generator.uniform(0.001, 0.1),
            "tumor_modulus": generator.uniform(3.0, 50.0),
            "liver_modulus": generator.uniform(0.4, 6.0),
            "poisson_ratio": generator.uniform(0.45, 0.49),
            "arterial_fraction": arterial,
            "portal_fraction": 1.0 - arterial,
            "tumor_axis_a_mm": generator.uniform(5.0, 25.0),
            "axis_ratio_b": generator.uniform(0.7, 1.3),
            "axis_ratio_c": generator.uniform(0.7, 1.3),
            "days": 180,
            "time_step_days": 1,
            "mesh_elements": 50000,
        }
        records.append(record)
    (output / "febio_parameters.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
