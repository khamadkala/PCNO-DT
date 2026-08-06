from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pcno-prepare")
    result.add_argument("--root", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    root = Path(arguments.root)
    patients = sorted(path for path in root.iterdir() if path.is_dir())
    fields = [
        "patient_id",
        "pre",
        "arterial",
        "portal",
        "delayed",
        "liver_mask",
        "tumor_mask",
        "trajectory",
        "time_to_progression",
        "event",
    ]
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for patient in patients:
            writer.writerow(
                {
                    "patient_id": patient.name,
                    "pre": str(patient / "pre.nii.gz"),
                    "arterial": str(patient / "arterial.nii.gz"),
                    "portal": str(patient / "portal.nii.gz"),
                    "delayed": str(patient / "delayed.nii.gz"),
                    "liver_mask": str(patient / "liver.nii.gz"),
                    "tumor_mask": str(patient / "tumor.nii.gz"),
                    "trajectory": str(patient / "trajectory.npz"),
                    "time_to_progression": "",
                    "event": "",
                }
            )


if __name__ == "__main__":
    main()

