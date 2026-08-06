from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pcno_dt.evaluation.statistics import bootstrap_interval, paired_comparison, student_interval


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pcno-evaluate")
    result.add_argument("--proposed", required=True)
    result.add_argument("--baseline")
    result.add_argument("--metric", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--bootstrap-samples", type=int, default=10000)
    return result


def main() -> None:
    arguments = parser().parse_args()
    proposed = np.loadtxt(arguments.proposed, delimiter=",", ndmin=1)
    student = student_interval(proposed)
    bootstrap = bootstrap_interval(proposed, samples=arguments.bootstrap_samples)
    report: dict[str, object] = {
        "metric": arguments.metric,
        "count": int(proposed.size),
        "mean": student.estimate,
        "student_95": [student.lower, student.upper],
        "bootstrap_95": [bootstrap.lower, bootstrap.upper],
    }
    if arguments.baseline:
        baseline = np.loadtxt(arguments.baseline, delimiter=",", ndmin=1)
        comparison = paired_comparison(proposed, baseline)
        report["comparison"] = comparison.__dict__
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

