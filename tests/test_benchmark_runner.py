#!/usr/bin/env python3

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools" / "run_benchmark_matrix.py"


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def write_fake_cli(path: Path) -> None:
    script = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def take_arg(flag):
    if flag in sys.argv:
        index = sys.argv.index(flag)
        return sys.argv[index + 1]
    return ""


query_file = take_arg("--query-file")
trace_jsonl = take_arg("--query-trace-jsonl-out")
summary_csv = take_arg("--query-summary-csv-out")
batch_report = take_arg("--query-batch-report-out")
snapshot_out = take_arg("--state-snapshot-out")

preset = "dense_only"
if "--adaptive-graph" in sys.argv:
    preset = "adaptive_graph"
elif "--lexical-prefilter" in sys.argv and "--semantic-hash-prefilter" in sys.argv:
    preset = "static_tiered"

metrics = {
    "dense_only": {"escalation_count": 0, "total_ms": 42.0, "coverage_ratio": 0.25, "peak_rss_kb": 1000},
    "static_tiered": {"escalation_count": 0, "total_ms": 30.0, "coverage_ratio": 0.55, "peak_rss_kb": 1200},
    "adaptive_graph": {"escalation_count": 1, "total_ms": 24.0, "coverage_ratio": 0.80, "peak_rss_kb": 1400},
}[preset]

queries = [line.strip() for line in Path(query_file).read_text(encoding="utf-8").splitlines() if line.strip()]
Path(trace_jsonl).write_text("".join(json.dumps({"query": query, "preset": preset}) + "\\n" for query in queries), encoding="utf-8")
Path(summary_csv).write_text("query,preset\\n" + "".join(f"\\"{query}\\",{preset}\\n" for query in queries), encoding="utf-8")
Path(snapshot_out).write_text("STATE_SNAPSHOT_V1\\n", encoding="utf-8")
Path(batch_report).write_text(json.dumps({
    "query_count": len(queries),
    "escalation_count": metrics["escalation_count"],
    "runtime": {
        "llm_backend": "FakeLLM",
        "embedding_backend": "FakeEmbedding",
        "llm_model_path": take_arg("--llm-model"),
        "embedding_model_path": take_arg("--embedding-model"),
        "sqlite_db_path": take_arg("--sqlite-db"),
        "index_path": take_arg("--index-path"),
        "query_source": "file",
        "num_threads": int(take_arg("--threads") or "4"),
        "max_new_tokens": int(take_arg("--max-new-tokens") or "256"),
        "sqlite_db_size_bytes": 12,
        "index_size_bytes": 34
    },
    "initial_graph_counts": {preset: len(queries)},
    "final_graph_counts": {preset: len(queries)},
    "fallback_reason_counts": {"none": len(queries)},
    "totals": {"promoted_to_hot": 0, "demoted_to_warm": 0},
    "maxima": {"max_peak_rss_kb": metrics["peak_rss_kb"]},
    "averages": {
        "top_score": 0.9,
        "score_margin": 0.5,
        "coverage_ratio": metrics["coverage_ratio"],
        "lexical_candidate_count": 2.0,
        "hash_candidate_count": 1.0,
        "dense_result_count": 1.0,
        "query_embedding_ms": 1.0,
        "retrieval_ms": 2.0,
        "evidence_ms": 3.0,
        "state_update_ms": 4.0,
        "prompt_build_ms": 5.0,
        "generation_ms": 6.0,
        "total_ms": metrics["total_ms"],
        "peak_rss_kb": metrics["peak_rss_kb"]
    }
}, indent=2), encoding="utf-8")
"""
    write_text(path, script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_command(command, cwd=None):
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def test_runs_matrix_and_writes_manifest(temp_root: Path) -> None:
    fake_cli = temp_root / "fake_mobile_rag_cli.py"
    write_fake_cli(fake_cli)

    query_file = temp_root / "queries.txt"
    write_text(query_file, "what is sqlite?\nwhat is faiss?\n")

    llm_model = temp_root / "fake.gguf"
    embedding_model = temp_root / "embedding-config.json"
    sqlite_db = temp_root / "rag.sqlite3"
    index_path = temp_root / "rag.faiss"
    state_snapshot_in = temp_root / "state.snapshot.tsv"
    for path in (llm_model, embedding_model, sqlite_db, index_path, state_snapshot_in):
        write_text(path, "stub\n")

    output_dir = temp_root / "benchmark"
    result = run_command(
        [
            sys.executable,
            str(RUNNER),
            "--binary",
            str(fake_cli),
            "--query-file",
            str(query_file),
            "--llm-model",
            str(llm_model),
            "--embedding-model",
            str(embedding_model),
            "--sqlite-db",
            str(sqlite_db),
            "--index-path",
            str(index_path),
            "--state-snapshot-in",
            str(state_snapshot_in),
            "--output-dir",
            str(output_dir),
            "--preset",
            "dense_only",
            "--preset",
            "static_tiered",
            "--preset",
            "adaptive_graph",
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["run_count"] == 3
    assert [run["preset"] for run in summary_json["runs"]] == [
        "dense_only",
        "static_tiered",
        "adaptive_graph",
    ]
    assert summary_json["runs"][2]["escalation_count"] == 1
    assert summary_json["runs"][2]["average_total_ms"] == 24.0

    summary_csv_lines = (output_dir / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert summary_csv_lines[0].startswith("preset,query_count,escalation_count")
    assert len(summary_csv_lines) == 4

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["shared_config"]["query_file"] == str(query_file)
    assert [run["preset"] for run in manifest["runs"]] == [
        "dense_only",
        "static_tiered",
        "adaptive_graph",
    ]
    assert manifest["runs"][1]["artifact_paths"]["batch_report"].endswith(
        "runs/02_static_tiered/batch-report.json"
    )


def test_replays_manifest_into_new_output_dir(temp_root: Path) -> None:
    fake_cli = temp_root / "fake_mobile_rag_cli.py"
    write_fake_cli(fake_cli)

    query_file = temp_root / "queries.txt"
    write_text(query_file, "what is sqlite?\n")

    llm_model = temp_root / "fake.gguf"
    embedding_model = temp_root / "embedding-config.json"
    sqlite_db = temp_root / "rag.sqlite3"
    index_path = temp_root / "rag.faiss"
    for path in (llm_model, embedding_model, sqlite_db, index_path):
        write_text(path, "stub\n")

    first_output_dir = temp_root / "first"
    first_run = run_command(
        [
            sys.executable,
            str(RUNNER),
            "--binary",
            str(fake_cli),
            "--query-file",
            str(query_file),
            "--llm-model",
            str(llm_model),
            "--embedding-model",
            str(embedding_model),
            "--sqlite-db",
            str(sqlite_db),
            "--index-path",
            str(index_path),
            "--output-dir",
            str(first_output_dir),
        ],
        cwd=REPO_ROOT,
    )
    assert first_run.returncode == 0, first_run.stderr or first_run.stdout

    replay_output_dir = temp_root / "replay"
    replay_run = run_command(
        [
            sys.executable,
            str(RUNNER),
            "--replay-manifest",
            str(first_output_dir / "manifest.json"),
            "--output-dir",
            str(replay_output_dir),
            "--binary",
            str(fake_cli),
        ],
        cwd=REPO_ROOT,
    )
    assert replay_run.returncode == 0, replay_run.stderr or replay_run.stdout

    replay_summary = json.loads((replay_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert replay_summary["run_count"] == 3
    assert replay_summary["runs"][0]["preset"] == "dense_only"
    assert (replay_output_dir / "runs" / "03_adaptive_graph" / "batch-report.json").exists()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_rag_benchmark_runner_") as temp_dir:
        temp_root = Path(temp_dir)
        test_runs_matrix_and_writes_manifest(temp_root / "case1")
        test_replays_manifest_into_new_output_dir(temp_root / "case2")
    print("Benchmark runner test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
