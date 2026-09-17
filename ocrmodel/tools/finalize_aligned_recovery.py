"""Publish the validation-only selection and its separate recovery verdict."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def finalize(run_dir: Path, smoke: bool = False) -> dict:
    read = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))
    summary, selection, metadata = read("summary.json"), read("selection.json"), read("metadata.json")
    if metadata["test_manifest_read"] or summary["test_used_for_selection"]:
        raise ValueError("training/selection must not read test")
    if summary["status"] != "complete" or not (run_dir / "COMPLETED").is_file():
        raise ValueError("training and checkpoint reloaded validation must complete")
    metrics = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text().splitlines() if line]
    if sum(row.get("aligned_rollout_pages", 0) for row in metrics) <= 0:
        raise ValueError("aligned recovery never collected a rollout")
    if smoke:
        result = {"status": "passed", "kind": "two_step_five_gpu_smoke",
                  "rollout_pages": sum(row.get("aligned_rollout_pages", 0) for row in metrics),
                  "accepted_pages": sum(row.get("aligned_accepted_pages", 0) for row in metrics),
                  "checkpoint_reloaded_validation": True}
    else:
        diagnostic = read("diagnostic_summary.json")
        baseline = next((p["validation"] for p in diagnostic["points"] if p["step"] == 0), None)
        selected = next(p for p in selection["candidates"] if p["step"] == selection["selected_step"])
        deletion_rate = lambda row: row["deletions"] / max(1, row["reference_characters"])
        if baseline is None:
            result = {"status": "selected_without_identity_baseline",
                      "identity_baseline_available": False,
                      "selected_step": selection["selected_step"], "baseline_cer": None,
                      "selected_cer": selected["cer"], "test_used_for_selection": False,
                      "historical_test_exposure": True, "automatic_retuning": False}
        else:
            checks = {
                "cer_improved": selected["cer"] < baseline["cer"],
                "loop_rate_improved": selected["loop_detected_page_rate"] < baseline["loop_detected_page_rate"],
                "length_limit_rate_improved": selected["generation_limit_hit_rate"] < baseline["generation_limit_hit_rate"],
                "deletion_increase_at_most_1pp": deletion_rate(selected) - deletion_rate(baseline) <= 0.01,
            }
            result = {"status": "passed" if all(checks.values()) else "not_passed", "checks": checks,
                      "identity_baseline_available": True,
                      "selected_step": selection["selected_step"], "baseline_cer": baseline["cer"],
                      "selected_cer": selected["cer"], "test_used_for_selection": False,
                      "historical_test_exposure": True, "automatic_retuning": False}
        group_selection = {**selection, "seeds": [42], "seed_runs": {
            "42": {"run_dir": str(run_dir), "selected_step": selection["selected_step"]}}}
        (run_dir.parent / "selection.json").write_text(json.dumps(group_selection, ensure_ascii=False, indent=2), encoding="utf-8")
    result["checkpoint_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (run_dir / f"checkpoint-{selection['selected_step']}").glob("*.safetensors")}
    (run_dir / "aligned_recovery_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    finalize(args.run_dir, args.smoke)
