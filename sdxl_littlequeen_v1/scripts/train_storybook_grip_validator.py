#!/usr/bin/env python3
"""Train and cross-validate a CLIP grip classifier from saved human reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from validate_storybook_accessories_v4 import grip_clip_components, grip_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_examples(run_root: Path) -> list[dict]:
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    records = {record["id"]: record for record in summary["records"]}
    review = json.loads(
        (run_root / "review" / "human_review.json").read_text(encoding="utf-8")
    )
    examples = []
    for decision in review["records"]:
        record = records[decision["id"]]
        artifacts = record["artifacts"]
        if "final" not in artifacts:
            continue
        examples.append(
            {
                "run": run_root.name,
                "id": decision["id"],
                "label": int(decision["decision"] == "accept"),
                "image": run_root / artifacts["final"],
                "metadata": run_root / artifacts["metadata"],
            }
        )
    return examples


def embeddings(examples: list[dict]) -> np.ndarray:
    processor, model, torch = grip_clip_components()
    crops = [grip_crop(item["image"], item["metadata"]) for item in examples]
    inputs = processor(images=crops, return_tensors="pt")
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    values = features.cpu().numpy().astype(np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def confusion(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positive = labels == 1
    return {
        "true_accept": int((predicted & positive).sum()),
        "true_reject": int((~predicted & ~positive).sum()),
        "false_accept": int((predicted & ~positive).sum()),
        "false_reject": int((~predicted & positive).sum()),
    }


def metrics(values: dict) -> dict:
    precision_denominator = values["true_accept"] + values["false_accept"]
    recall_denominator = values["true_accept"] + values["false_reject"]
    total = sum(values.values())
    return {
        "agreement": round(
            (values["true_accept"] + values["true_reject"]) / max(total, 1),
            6,
        ),
        "accept_precision": round(
            values["true_accept"] / max(precision_denominator, 1),
            6,
        ),
        "accept_recall": round(
            values["true_accept"] / max(recall_denominator, 1),
            6,
        ),
        "confusion": values,
    }


def make_model() -> LogisticRegression:
    return LogisticRegression(
        C=0.2,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=0,
    )


def main() -> int:
    args = parse_args()
    examples = [
        example
        for run_root in args.run_root
        for example in load_examples(run_root.resolve())
    ]
    features = embeddings(examples)
    labels = np.asarray([item["label"] for item in examples], dtype=np.int64)
    runs = np.asarray([item["run"] for item in examples])
    probabilities = np.zeros(len(examples), dtype=np.float64)
    folds = []
    for held_out in sorted(set(runs)):
        train = runs != held_out
        test = ~train
        model = make_model().fit(features[train], labels[train])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
        folds.append(
            {
                "held_out_run": held_out,
                "example_count": int(test.sum()),
                **metrics(confusion(labels[test], probabilities[test], 0.5)),
            }
        )

    threshold_candidates = np.linspace(0.25, 0.8, 112)
    ranked = []
    for threshold in threshold_candidates:
        result = metrics(confusion(labels, probabilities, float(threshold)))
        ranked.append(
            (
                result["accept_precision"] >= 0.8,
                result["accept_recall"],
                result["accept_precision"],
                result["agreement"],
                -float(threshold),
                float(threshold),
                result,
            )
        )
    _, _, _, _, _, threshold, cross_validated = max(ranked)
    threshold = round(threshold, 6)
    final_model = make_model().fit(features, labels)
    artifact = {
        "version": "littlequeen-grip-clip-linear-v1",
        "clip_model": "openai/clip-vit-base-patch32",
        "feature_dimension": int(features.shape[1]),
        "threshold": threshold,
        "coefficient": [round(float(value), 10) for value in final_model.coef_[0]],
        "intercept": round(float(final_model.intercept_[0]), 10),
    }
    report = {
        "version": artifact["version"],
        "training_example_count": len(examples),
        "positive_count": int(labels.sum()),
        "runs": sorted(set(runs)),
        "threshold": threshold,
        "leave_one_run_out": {
            **cross_validated,
            "folds_at_0_5": folds,
            "predictions": [
                {
                    "run": item["run"],
                    "id": item["id"],
                    "human": "accept" if item["label"] else "reject",
                    "probability": round(float(probability), 6),
                }
                for item, probability in zip(examples, probabilities, strict=True)
            ],
        },
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "leave_one_run_out"}, indent=2))
    print(json.dumps(report["leave_one_run_out"] | {"predictions": "omitted"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
