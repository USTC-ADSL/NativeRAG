#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


PRESET_FLAGS: Dict[str, List[str]] = {
    "dense_only": [],
    "static_tiered": ["--lexical-prefilter", "--semantic-hash-prefilter"],
    "adaptive_graph": [
        "--lexical-prefilter",
        "--semantic-hash-prefilter",
        "--adaptive-graph",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run query-file benchmark ablations around the existing mobile_rag_cli "
            "trace, batch report, and snapshot exports."
        )
    )
    parser.add_argument("--binary", help="Path to the query CLI binary")
    parser.add_argument("--query-file", help="Path to the batch query file")
    parser.add_argument("--llm-model", help="Path to the LLM model")
    parser.add_argument("--embedding-model", help="Path to the embedding model")
    parser.add_argument("--sqlite-db", help="Path to the SQLite database")
    parser.add_argument("--index-path", help="Path to the Faiss index")
    parser.add_argument("--state-snapshot-in", help="Optional chunk-state snapshot input")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will receive run bundles, summaries, and the manifest",
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Ablation preset to run. Defaults to dense_only, static_tiered, adaptive_graph",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lexical-candidates", type=int, default=16)
    parser.add_argument("--semantic-hash-candidates", type=int, default=32)
    parser.add_argument("--semantic-hash-max-distance", type=int, default=-1)
    parser.add_argument(
        "--replay-manifest",
        help="Replay a previously saved benchmark manifest into a fresh output directory",
    )
    return parser.parse_args()


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def ensure_path_exists(path_str: str, label: str) -> str:
    if not path_str:
        fail(f"Error: {label} is required")
    path = Path(path_str)
    if not path.exists():
        fail(f"Error: {label} not found: {path}")
    return str(path.resolve())


def resolve_optional_path(path_str: str | None) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists():
        fail(f"Error: optional path not found: {path}")
    return str(path.resolve())


def normalize_presets(presets: Iterable[str] | None) -> List[str]:
    normalized = list(presets or ["dense_only", "static_tiered", "adaptive_graph"])
    if not normalized:
        fail("Error: at least one preset is required")
    invalid = [preset for preset in normalized if preset not in PRESET_FLAGS]
    if invalid:
        fail(
            "Error: unsupported preset(s): "
            + ", ".join(invalid)
            + ". Supported presets: "
            + ", ".join(PRESET_FLAGS.keys())
        )
    return normalized


def build_shared_config(args: argparse.Namespace) -> dict:
    return {
        "binary": ensure_path_exists(args.binary, "--binary"),
        "query_file": ensure_path_exists(args.query_file, "--query-file"),
        "llm_model": ensure_path_exists(args.llm_model, "--llm-model"),
        "embedding_model": ensure_path_exists(args.embedding_model, "--embedding-model"),
        "sqlite_db": ensure_path_exists(args.sqlite_db, "--sqlite-db"),
        "index_path": ensure_path_exists(args.index_path, "--index-path"),
        "state_snapshot_in": resolve_optional_path(args.state_snapshot_in),
        "top_k": args.top_k,
        "threads": args.threads,
        "max_new_tokens": args.max_new_tokens,
        "lexical_candidates": args.lexical_candidates,
        "semantic_hash_candidates": args.semantic_hash_candidates,
        "semantic_hash_max_distance": args.semantic_hash_max_distance,
        "presets": normalize_presets(args.presets),
    }


def load_shared_config_from_manifest(manifest_path: Path, binary_override: str | None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        fail(f"Error: unsupported manifest schema version in {manifest_path}")
    shared_config = dict(manifest["shared_config"])
    shared_config["presets"] = normalize_presets(shared_config.get("presets"))
    if binary_override:
        shared_config["binary"] = ensure_path_exists(binary_override, "--binary")
    else:
        shared_config["binary"] = ensure_path_exists(shared_config["binary"], "manifest binary")
    shared_config["query_file"] = ensure_path_exists(shared_config["query_file"], "manifest query_file")
    shared_config["llm_model"] = ensure_path_exists(shared_config["llm_model"], "manifest llm_model")
    shared_config["embedding_model"] = ensure_path_exists(
        shared_config["embedding_model"], "manifest embedding_model"
    )
    shared_config["sqlite_db"] = ensure_path_exists(shared_config["sqlite_db"], "manifest sqlite_db")
    shared_config["index_path"] = ensure_path_exists(shared_config["index_path"], "manifest index_path")
    shared_config["state_snapshot_in"] = resolve_optional_path(shared_config.get("state_snapshot_in"))
    return shared_config


def make_run_artifacts(run_dir: Path) -> dict:
    return {
        "trace_jsonl": run_dir / "query-trace.jsonl",
        "summary_csv": run_dir / "query-summary.csv",
        "batch_report": run_dir / "batch-report.json",
        "state_snapshot_out": run_dir / "state.snapshot.tsv",
        "stdout": run_dir / "stdout.log",
        "stderr": run_dir / "stderr.log",
    }


def build_command(shared_config: dict, preset: str, artifacts: dict) -> List[str]:
    command = [
        shared_config["binary"],
        "--query",
        "--query-file",
        shared_config["query_file"],
        "--llm-model",
        shared_config["llm_model"],
        "--embedding-model",
        shared_config["embedding_model"],
        "--sqlite-db",
        shared_config["sqlite_db"],
        "--index-path",
        shared_config["index_path"],
        "--top-k",
        str(shared_config["top_k"]),
        "--threads",
        str(shared_config["threads"]),
        "--max-new-tokens",
        str(shared_config["max_new_tokens"]),
        "--query-trace-jsonl-out",
        str(artifacts["trace_jsonl"]),
        "--query-summary-csv-out",
        str(artifacts["summary_csv"]),
        "--query-batch-report-out",
        str(artifacts["batch_report"]),
        "--state-snapshot-out",
        str(artifacts["state_snapshot_out"]),
    ]
    if shared_config.get("state_snapshot_in"):
        command.extend(["--state-snapshot-in", shared_config["state_snapshot_in"]])
    if PRESET_FLAGS[preset]:
        command.extend(
            [
                "--lexical-candidates",
                str(shared_config["lexical_candidates"]),
                "--semantic-hash-candidates",
                str(shared_config["semantic_hash_candidates"]),
                "--semantic-hash-max-distance",
                str(shared_config["semantic_hash_max_distance"]),
            ]
        )
    command.extend(PRESET_FLAGS[preset])
    return command


def run_matrix(shared_config: dict, output_dir: Path) -> tuple[dict, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary_runs = []
    manifest_runs = []

    for index, preset in enumerate(shared_config["presets"], start=1):
        run_dir = runs_dir / f"{index:02d}_{preset}"
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = make_run_artifacts(run_dir)
        command = build_command(shared_config, preset, artifacts)
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        artifacts["stdout"].write_text(result.stdout, encoding="utf-8")
        artifacts["stderr"].write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            fail(
                f"Error: benchmark preset '{preset}' failed with exit code {result.returncode}. "
                f"See {artifacts['stderr']}"
            )

        batch_report = json.loads(artifacts["batch_report"].read_text(encoding="utf-8"))
        summary_runs.append(
            {
                "preset": preset,
                "query_count": batch_report["query_count"],
                "escalation_count": batch_report["escalation_count"],
                "average_total_ms": batch_report["averages"]["total_ms"],
                "average_coverage_ratio": batch_report["averages"]["coverage_ratio"],
                "max_peak_rss_kb": batch_report["maxima"]["max_peak_rss_kb"],
                "batch_report_path": artifacts["batch_report"].relative_to(output_dir).as_posix(),
                "trace_jsonl_path": artifacts["trace_jsonl"].relative_to(output_dir).as_posix(),
                "summary_csv_path": artifacts["summary_csv"].relative_to(output_dir).as_posix(),
                "state_snapshot_out_path": artifacts["state_snapshot_out"]
                .relative_to(output_dir)
                .as_posix(),
            }
        )
        manifest_runs.append(
            {
                "preset": preset,
                "preset_flags": PRESET_FLAGS[preset],
                "command": command,
                "artifact_paths": {
                    key: value.relative_to(output_dir).as_posix()
                    for key, value in artifacts.items()
                },
            }
        )

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(summary_runs),
        "runs": summary_runs,
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shared_config": shared_config,
        "runs": manifest_runs,
    }
    return summary, manifest


def write_summary_csv(summary: dict, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "preset",
                "query_count",
                "escalation_count",
                "average_total_ms",
                "average_coverage_ratio",
                "max_peak_rss_kb",
                "batch_report_path",
                "trace_jsonl_path",
                "summary_csv_path",
                "state_snapshot_out_path",
            ]
        )
        for run in summary["runs"]:
            writer.writerow(
                [
                    run["preset"],
                    run["query_count"],
                    run["escalation_count"],
                    run["average_total_ms"],
                    run["average_coverage_ratio"],
                    run["max_peak_rss_kb"],
                    run["batch_report_path"],
                    run["trace_jsonl_path"],
                    run["summary_csv_path"],
                    run["state_snapshot_out_path"],
                ]
            )


def write_outputs(summary: dict, manifest: dict, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary_csv(summary, output_dir / "summary.csv")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.replay_manifest:
        manifest_path = Path(args.replay_manifest).resolve()
        if not manifest_path.exists():
            fail(f"Error: replay manifest not found: {manifest_path}")
        shared_config = load_shared_config_from_manifest(manifest_path, args.binary)
    else:
        shared_config = build_shared_config(args)

    summary, manifest = run_matrix(shared_config, output_dir)
    write_outputs(summary, manifest, output_dir)
    print(f"Wrote benchmark summary to {output_dir / 'summary.json'}")
    print(f"Wrote benchmark manifest to {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
