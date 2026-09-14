#!/usr/bin/env python3
"""Compare automatic accessory validation with a human review file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    automatic_data = json.loads(args.automatic.read_text(encoding="utf-8"))
    human_data = json.loads(args.human.read_text(encoding="utf-8"))
    automatic = {
        record["id"]: bool(record["validation"]["accepted"])
        for record in automatic_data["records"]
    }
    human = {record["id"]: record["decision"] == "accept" for record in human_data["records"]}
    ids = sorted(set(automatic) & set(human))
    confusion = {"true_accept": 0, "true_reject": 0, "false_accept": 0, "false_reject": 0}
    mismatches = []
    for item_id in ids:
        auto = automatic[item_id]
        person = human[item_id]
        if auto and person:
            confusion["true_accept"] += 1
        elif not auto and not person:
            confusion["true_reject"] += 1
        elif auto:
            confusion["false_accept"] += 1
            mismatches.append({"id": item_id, "automatic": "accept", "human": "reject"})
        else:
            confusion["false_reject"] += 1
            mismatches.append({"id": item_id, "automatic": "reject", "human": "accept"})
    agreement = (
        (confusion["true_accept"] + confusion["true_reject"]) / len(ids) if ids else 0.0
    )
    accepted_by_automatic = confusion["true_accept"] + confusion["false_accept"]
    accepted_by_human = confusion["true_accept"] + confusion["false_reject"]
    rejected_by_human = confusion["true_reject"] + confusion["false_accept"]
    report = {
        "automatic_validator_version": automatic_data["validator_version"],
        "reviewed_count": len(ids),
        "agreement": round(agreement, 6),
        "accept_precision": round(
            confusion["true_accept"] / accepted_by_automatic,
            6,
        )
        if accepted_by_automatic
        else 0.0,
        "accept_recall": round(
            confusion["true_accept"] / accepted_by_human,
            6,
        )
        if accepted_by_human
        else 0.0,
        "reject_recall": round(
            confusion["true_reject"] / rejected_by_human,
            6,
        )
        if rejected_by_human
        else 0.0,
        "confusion": confusion,
        "mismatches": mismatches,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
