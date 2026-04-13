#!/usr/bin/env python3

import json
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


def write_fake_cli(
    path: Path,
    require_binary_dir_in_ld_library_path: bool = False,
    require_cwd_matches_binary_dir: bool = False,
) -> None:
    script = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


REQUIRE_BINARY_DIR_IN_LD_LIBRARY_PATH = __REQUIRE_BINARY_DIR_IN_LD_LIBRARY_PATH__
REQUIRE_CWD_MATCHES_BINARY_DIR = __REQUIRE_CWD_MATCHES_BINARY_DIR__


def take_arg(flag):
    if flag in sys.argv:
        index = sys.argv.index(flag)
        return sys.argv[index + 1]
    return ""


if REQUIRE_BINARY_DIR_IN_LD_LIBRARY_PATH:
    binary_dir = str(Path(sys.argv[0]).resolve().parent)
    library_path_entries = []
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not entry:
            continue
        entry_path = Path(entry)
        if not entry_path.is_absolute():
            entry_path = (Path.cwd() / entry_path).resolve()
        library_path_entries.append(str(entry_path))
    if binary_dir not in library_path_entries:
        sys.stderr.write(
            f'CANNOT LINK EXECUTABLE "{{sys.argv[0]}}": library "libMNN.so" not found\\n'
        )
        sys.exit(1)

if REQUIRE_CWD_MATCHES_BINARY_DIR:
    binary_dir = Path(sys.argv[0]).resolve().parent
    if Path.cwd().resolve() != binary_dir:
        sys.stderr.write(
            f'Expected cwd "{{Path.cwd()}}" to match binary dir "{{binary_dir}}"\\n'
        )
        sys.exit(1)


query_file = take_arg("--query-file")
trace_jsonl = take_arg("--query-trace-jsonl-out")
summary_csv = take_arg("--query-summary-csv-out")
batch_report = take_arg("--query-batch-report-out")
snapshot_out = take_arg("--state-snapshot-out")

preset = "dense_only"
if "--adaptive-graph" in sys.argv and "--state-aware-dense" in sys.argv:
    preset = "adaptive_state_aware"
elif "--adaptive-graph" in sys.argv:
    preset = "adaptive_graph"
elif (
    "--lexical-prefilter" in sys.argv
    and "--semantic-hash-prefilter" in sys.argv
    and "--state-aware-dense" in sys.argv
):
    preset = "state_aware_tiered"
elif "--lexical-prefilter" in sys.argv and "--semantic-hash-prefilter" in sys.argv:
    preset = "static_tiered"
elif "--state-aware-dense" in sys.argv:
    preset = "dense_only_state_aware"

metrics = {
    "dense_only": {"escalation_count": 0, "total_ms": 42.0, "coverage_ratio": 0.25, "peak_rss_kb": 1000, "p50_total_ms": 40.0, "p95_total_ms": 44.0, "state_aware_dense_query_count": 0, "state_filtered_candidate_count": 0.0},
    "dense_only_state_aware": {"escalation_count": 0, "total_ms": 39.0, "coverage_ratio": 0.35, "peak_rss_kb": 980, "p50_total_ms": 38.0, "p95_total_ms": 41.0, "state_aware_dense_query_count": 1, "state_filtered_candidate_count": 2.0},
    "static_tiered": {"escalation_count": 0, "total_ms": 30.0, "coverage_ratio": 0.55, "peak_rss_kb": 1200, "p50_total_ms": 29.0, "p95_total_ms": 31.0, "state_aware_dense_query_count": 0, "state_filtered_candidate_count": 0.0},
    "state_aware_tiered": {"escalation_count": 0, "total_ms": 27.0, "coverage_ratio": 0.63, "peak_rss_kb": 1180, "p50_total_ms": 26.0, "p95_total_ms": 29.0, "state_aware_dense_query_count": 1, "state_filtered_candidate_count": 1.5},
    "adaptive_graph": {"escalation_count": 1, "total_ms": 24.0, "coverage_ratio": 0.80, "peak_rss_kb": 1400, "p50_total_ms": 23.0, "p95_total_ms": 25.0, "state_aware_dense_query_count": 0, "state_filtered_candidate_count": 0.0},
    "adaptive_state_aware": {"escalation_count": 1, "total_ms": 22.0, "coverage_ratio": 0.84, "peak_rss_kb": 1350, "p50_total_ms": 21.0, "p95_total_ms": 23.0, "state_aware_dense_query_count": 1, "state_filtered_candidate_count": 1.0},
}[preset]

queries = [line.strip() for line in Path(query_file).read_text(encoding="utf-8").splitlines() if line.strip()]
Path(trace_jsonl).write_text("".join(json.dumps({"query": query, "preset": preset}) + "\\n" for query in queries), encoding="utf-8")
Path(summary_csv).write_text("query,preset\\n" + "".join(f"\\"{query}\\",{preset}\\n" for query in queries), encoding="utf-8")
Path(snapshot_out).write_text("STATE_SNAPSHOT_V1\\n", encoding="utf-8")
Path(batch_report).write_text(json.dumps({
    "query_count": len(queries),
    "escalation_count": metrics["escalation_count"],
    "state_aware_dense_query_count": metrics["state_aware_dense_query_count"],
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
    "percentiles": {
        "p50": {"total_ms": metrics["p50_total_ms"], "peak_rss_kb": metrics["peak_rss_kb"] - 50},
        "p95": {"total_ms": metrics["p95_total_ms"], "peak_rss_kb": metrics["peak_rss_kb"] + 25}
    },
    "averages": {
        "top_score": 0.9,
        "score_margin": 0.5,
        "coverage_ratio": metrics["coverage_ratio"],
        "lexical_candidate_count": 2.0,
        "hash_candidate_count": 1.0,
        "state_filtered_candidate_count": metrics["state_filtered_candidate_count"],
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
    script = script.replace(
        "__REQUIRE_BINARY_DIR_IN_LD_LIBRARY_PATH__",
        repr(require_binary_dir_in_ld_library_path),
    )
    script = script.replace(
        "__REQUIRE_CWD_MATCHES_BINARY_DIR__",
        repr(require_cwd_matches_binary_dir),
    )
    write_text(path, script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def write_fake_adb(path: Path, device_root: Path) -> None:
    script = f"""#!/usr/bin/env python3
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


DEVICE_ROOT = Path({device_root.as_posix()!r})


def to_local(remote_path):
    return DEVICE_ROOT / remote_path.lstrip("/")


args = sys.argv[1:]
if args[:1] == ["-s"]:
    args = args[2:]

if args == ["devices"]:
    sys.stdout.write("List of devices attached\\nfd8657d6\\tdevice\\n")
    sys.exit(0)

if args == ["get-state"]:
    sys.stdout.write("device\\n")
    sys.exit(0)

if not args:
    sys.exit(1)

if args[0] == "shell":
    shell_args = args[1:]
    if shell_args[:2] == ["mkdir", "-p"]:
        to_local(shell_args[2]).mkdir(parents=True, exist_ok=True)
        sys.exit(0)
    if len(shell_args) == 1:
        command_string = shell_args[0]
        segments = [shlex.split(part.strip()) for part in command_string.split("&&")]
        cwd = None
        env = os.environ.copy()
        translated = []
        for segment in segments:
            if not segment:
                continue
            if segment[0] == "cd":
                target = segment[1]
                cwd = to_local(target) if target.startswith("/") else Path(target)
                continue
            if segment[0] == "env":
                command_tokens = segment[1:]
                while command_tokens and "=" in command_tokens[0]:
                    key, value = command_tokens.pop(0).split("=", 1)
                    translated_paths = []
                    for entry in value.split(":"):
                        if entry.startswith("/"):
                            translated_paths.append(str(to_local(entry)))
                        else:
                            translated_paths.append(entry)
                    env[key] = ":".join(translated_paths)
                translated = command_tokens
                break
            translated = segment
        normalized = []
        for token in translated:
            if token.startswith("/"):
                normalized.append(str(to_local(token)))
            else:
                normalized.append(token)
        result = subprocess.run(
            normalized,
            text=True,
            capture_output=True,
            check=False,
            cwd=cwd,
            env=env,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    translated = []
    for token in shell_args:
        if token.startswith("LD_LIBRARY_PATH="):
            key, value = token.split("=", 1)
            translated_paths = []
            for entry in value.split(":"):
                if entry.startswith("/"):
                    translated_paths.append(str(to_local(entry)))
                else:
                    translated_paths.append(entry)
            translated.append(key + "=" + ":".join(translated_paths))
        elif token.startswith("/"):
            translated.append(str(to_local(token)))
        else:
            translated.append(token)
    result = subprocess.run(translated, text=True, capture_output=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)

if args[0] == "pull":
    source = to_local(args[1])
    destination = Path(args[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    sys.stdout.write(f"{{args[1]}}: 1 file pulled\\n")
    sys.exit(0)

sys.exit(1)
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
    assert summary_json["runs"][0]["p50_total_ms"] == 40.0
    assert summary_json["runs"][2]["p95_total_ms"] == 25.0

    summary_csv_lines = (output_dir / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert summary_csv_lines[0].startswith(
        "preset,query_count,escalation_count,state_aware_dense_query_count,p50_total_ms,p95_total_ms"
    )
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


def test_runs_matrix_via_fake_adb_device(temp_root: Path) -> None:
    device_root = temp_root / "device_root"
    remote_binary = device_root / "remote" / "bin" / "mobile_rag_cli"
    write_fake_cli(remote_binary)

    remote_query_file = device_root / "remote" / "input" / "queries.txt"
    write_text(remote_query_file, "what is sqlite?\nwhat is faiss?\n")

    for remote_path in (
        device_root / "remote" / "models" / "fake.gguf",
        device_root / "remote" / "models" / "embedding-config.json",
        device_root / "remote" / "runtime" / "rag.sqlite3",
        device_root / "remote" / "runtime" / "rag.faiss",
        device_root / "remote" / "runtime" / "state.snapshot.tsv",
    ):
        write_text(remote_path, "stub\n")

    fake_adb = temp_root / "fake_adb.py"
    write_fake_adb(fake_adb, device_root)

    output_dir = temp_root / "device_benchmark"
    result = run_command(
        [
            sys.executable,
            str(RUNNER),
            "--adb",
            str(fake_adb),
            "--adb-serial",
            "fd8657d6",
            "--remote-workdir",
            "/remote/bench",
            "--binary",
            "/remote/bin/mobile_rag_cli",
            "--query-file",
            "/remote/input/queries.txt",
            "--llm-model",
            "/remote/models/fake.gguf",
            "--embedding-model",
            "/remote/models/embedding-config.json",
            "--sqlite-db",
            "/remote/runtime/rag.sqlite3",
            "--index-path",
            "/remote/runtime/rag.faiss",
            "--state-snapshot-in",
            "/remote/runtime/state.snapshot.tsv",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["run_count"] == 3
    assert (output_dir / "runs" / "01_dense_only" / "batch-report.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shared_config"]["execution_mode"] == "adb"
    assert manifest["shared_config"]["adb_serial"] == "fd8657d6"
    assert manifest["shared_config"]["remote_workdir"] == "/remote/bench"


def test_runs_matrix_via_fake_adb_device_with_binary_dir_library_path(temp_root: Path) -> None:
    device_root = temp_root / "device_root"
    remote_binary = device_root / "remote" / "bin" / "mobile_rag_cli"
    write_fake_cli(remote_binary, require_binary_dir_in_ld_library_path=True)

    remote_query_file = device_root / "remote" / "input" / "queries.txt"
    write_text(remote_query_file, "what is sqlite?\n")

    for remote_path in (
        device_root / "remote" / "models" / "fake.gguf",
        device_root / "remote" / "models" / "embedding-config.json",
        device_root / "remote" / "runtime" / "rag.sqlite3",
        device_root / "remote" / "runtime" / "rag.faiss",
        device_root / "remote" / "runtime" / "state.snapshot.tsv",
    ):
        write_text(remote_path, "stub\n")

    fake_adb = temp_root / "fake_adb.py"
    write_fake_adb(fake_adb, device_root)

    output_dir = temp_root / "device_benchmark_ld_library_path"
    result = run_command(
        [
            sys.executable,
            str(RUNNER),
            "--adb",
            str(fake_adb),
            "--adb-serial",
            "fd8657d6",
            "--remote-workdir",
            "/remote/bench",
            "--binary",
            "/remote/bin/mobile_rag_cli",
            "--query-file",
            "/remote/input/queries.txt",
            "--llm-model",
            "/remote/models/fake.gguf",
            "--embedding-model",
            "/remote/models/embedding-config.json",
            "--sqlite-db",
            "/remote/runtime/rag.sqlite3",
            "--index-path",
            "/remote/runtime/rag.faiss",
            "--state-snapshot-in",
            "/remote/runtime/state.snapshot.tsv",
            "--output-dir",
            str(output_dir),
            "--preset",
            "dense_only",
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_runs_matrix_via_fake_adb_device_from_binary_dir(temp_root: Path) -> None:
    device_root = temp_root / "device_root"
    remote_binary = device_root / "remote" / "bin" / "mobile_rag_cli"
    write_fake_cli(
        remote_binary,
        require_binary_dir_in_ld_library_path=True,
        require_cwd_matches_binary_dir=True,
    )

    remote_query_file = device_root / "remote" / "input" / "queries.txt"
    write_text(remote_query_file, "what is sqlite?\n")

    for remote_path in (
        device_root / "remote" / "models" / "fake.gguf",
        device_root / "remote" / "models" / "embedding-config.json",
        device_root / "remote" / "runtime" / "rag.sqlite3",
        device_root / "remote" / "runtime" / "rag.faiss",
        device_root / "remote" / "runtime" / "state.snapshot.tsv",
    ):
        write_text(remote_path, "stub\n")

    fake_adb = temp_root / "fake_adb.py"
    write_fake_adb(fake_adb, device_root)

    output_dir = temp_root / "device_benchmark_workdir"
    result = run_command(
        [
            sys.executable,
            str(RUNNER),
            "--adb",
            str(fake_adb),
            "--adb-serial",
            "fd8657d6",
            "--remote-workdir",
            "/remote/bench",
            "--binary",
            "/remote/bin/mobile_rag_cli",
            "--query-file",
            "/remote/input/queries.txt",
            "--llm-model",
            "/remote/models/fake.gguf",
            "--embedding-model",
            "/remote/models/embedding-config.json",
            "--sqlite-db",
            "/remote/runtime/rag.sqlite3",
            "--index-path",
            "/remote/runtime/rag.faiss",
            "--state-snapshot-in",
            "/remote/runtime/state.snapshot.tsv",
            "--output-dir",
            str(output_dir),
            "--preset",
            "dense_only",
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_accepts_state_aware_presets(temp_root: Path) -> None:
    fake_cli = temp_root / "fake_mobile_rag_cli.py"
    write_fake_cli(fake_cli)

    query_file = temp_root / "queries.txt"
    write_text(query_file, "what is sqlite?\n")

    llm_model = temp_root / "fake.gguf"
    embedding_model = temp_root / "embedding-config.json"
    sqlite_db = temp_root / "rag.sqlite3"
    index_path = temp_root / "rag.faiss"
    state_snapshot_in = temp_root / "state.snapshot.tsv"
    for path in (llm_model, embedding_model, sqlite_db, index_path, state_snapshot_in):
        write_text(path, "stub\n")

    output_dir = temp_root / "benchmark_state_aware"
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
            "dense_only_state_aware",
            "--preset",
            "state_aware_tiered",
            "--preset",
            "adaptive_state_aware",
        ],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["run_count"] == 3
    assert [run["preset"] for run in summary_json["runs"]] == [
        "dense_only_state_aware",
        "state_aware_tiered",
        "adaptive_state_aware",
    ]
    assert summary_json["runs"][0]["average_total_ms"] == 39.0
    assert summary_json["runs"][1]["average_coverage_ratio"] == 0.63
    assert summary_json["runs"][2]["p50_total_ms"] == 21.0
    assert summary_json["runs"][0]["state_aware_dense_query_count"] == 1
    assert summary_json["runs"][1]["average_state_filtered_candidate_count"] == 1.5

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"][0]["preset_flags"] == ["--state-aware-dense"]
    assert manifest["runs"][1]["preset_flags"] == [
        "--lexical-prefilter",
        "--semantic-hash-prefilter",
        "--state-aware-dense",
    ]
    assert manifest["runs"][2]["preset_flags"] == [
        "--lexical-prefilter",
        "--semantic-hash-prefilter",
        "--adaptive-graph",
        "--state-aware-dense",
    ]
    assert "--state-aware-dense" in manifest["runs"][2]["command"]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="native_rag_benchmark_runner_") as temp_dir:
        temp_root = Path(temp_dir)
        test_runs_matrix_and_writes_manifest(temp_root / "case1")
        test_replays_manifest_into_new_output_dir(temp_root / "case2")
        test_runs_matrix_via_fake_adb_device(temp_root / "case3")
        test_runs_matrix_via_fake_adb_device_with_binary_dir_library_path(temp_root / "case4")
        test_runs_matrix_via_fake_adb_device_from_binary_dir(temp_root / "case5")
        test_accepts_state_aware_presets(temp_root / "case6")
    print("Benchmark runner test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
