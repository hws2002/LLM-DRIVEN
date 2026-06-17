"""English documentation."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MACRO_DIR = PROJECT_ROOT / "macro"
MACRO_SRC = MACRO_DIR / "src"
MOCK_INPUT = PROJECT_ROOT / "input_data" / "mock" / "mock_data.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    sys.exit(1)


def _mb(bytes_: int) -> str:
    return f"{bytes_ / 1024 / 1024:.1f} MB"


def _snapshot(label: str) -> dict:
    proc = psutil.Process()
    mem = proc.memory_info()
    rss = mem.rss
    print(f"  [{label}] RSS={_mb(rss)}")
    return {"label": label, "rss": rss, "time": time.time()}


def measure_model_load() -> None:
    """English documentation."""
    print("\n" + "=" * 60)
    print("Model Load Memory Test")
    print("=" * 60)

    _snapshot("before import")

    from sentence_transformers import SentenceTransformer
    _snapshot("after sentence_transformers import")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\n  Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    s2 = _snapshot("after model load")

    from keybert import KeyBERT
    kb = KeyBERT(model=model)
    s3 = _snapshot("after KeyBERT init")

    print(f"\n  >> text text text text text: {_mb(s3['rss'] - s2['rss'] + (s2['rss'] - s2['rss']))}")
    print(f"  >> text text (text text text): {_mb(s3['rss'])}")


async def _run_subprocess(cmd: list, label: str) -> tuple[int, float]:
    """English documentation."""
    proc_info = psutil.Process()
    mem_before = proc_info.memory_info().rss
    t_start = time.time()

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(MACRO_DIR),
    )

    peak_child_rss = 0
    pid = process.pid

    async def drain(stream):
        while True:
            line = await stream.readline()
            if not line:
                break

    async def poll_memory():
        nonlocal peak_child_rss
        while process.returncode is None:
            try:
                child = psutil.Process(pid)
                rss = child.memory_info().rss
                # English comment.
                for grandchild in child.children(recursive=True):
                    try:
                        rss += grandchild.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if rss > peak_child_rss:
                    peak_child_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            await asyncio.sleep(0.5)

    await asyncio.gather(
        drain(process.stdout),
        drain(process.stderr),
        poll_memory(),
        process.wait(),
    )

    elapsed = time.time() - t_start
    print(f"  [{label}] peak child RSS = {_mb(peak_child_rss)},  elapsed = {elapsed:.1f}s,  returncode = {process.returncode}")
    return peak_child_rss, elapsed


async def measure_subprocess(concurrency: int, mock_input: Path) -> None:
    """English documentation."""
    print("\n" + "=" * 60)
    print(f"Subprocess Memory Test  (concurrency={concurrency})")
    print("=" * 60)

    if not mock_input.exists():
        print(f"ERROR: mock input not found: {mock_input}")
        print("  --input text text text text text.")
        sys.exit(1)

    from shared.env_loader import load_root_env
    from shared import config as cfg
    load_root_env()

    script = MACRO_SRC / "run_pipeline.py"
    config_path = MACRO_DIR / "config.yaml"
    output_dir = PROJECT_ROOT / "output_data" / "mem_check_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(MACRO_SRC), str(PROJECT_ROOT)]),
    }

    def make_cmd(idx: int) -> list:
        out = output_dir / f"run_{idx}"
        out.mkdir(exist_ok=True)
        return [
            sys.executable, "-u", str(script),
            "--input", str(mock_input),
            "--config", str(config_path),
            "--output-dir", str(out),
            "--provider", cfg.MACRO_LLM_PROVIDER,
            "--model", cfg.MACRO_LLM_MODEL,
            "--skip-indexing",
            "--no-llm-edges",
            "--min-clusters", "2",
            "--max-clusters", "5",
        ]

    parent = psutil.Process()
    mem_before = parent.memory_info().rss
    print(f"\n  parent RSS before: {_mb(mem_before)}")

    t0 = time.time()
    tasks = [_run_subprocess(make_cmd(i), f"subprocess-{i}") for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    total_elapsed = time.time() - t0

    mem_after = parent.memory_info().rss
    total_child_peak = sum(r[0] for r in results)

    print(f"\n{'=' * 60}")
    print(f"Results (concurrency={concurrency})")
    print(f"  parent RSS after  : {_mb(mem_after)}")
    print(f"  sum of child peak : {_mb(total_child_peak)}")
    print(f"  estimated total   : {_mb(mem_after + total_child_peak)}")
    print(f"  total wall time   : {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="macro subprocess text text")
    parser.add_argument("--mode", choices=["model", "subprocess", "all"], default="model",
                        help="model: text text / subprocess: text text / all: text text")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="text subprocess text (subprocess/all text)")
    parser.add_argument("--input", type=Path, default=MOCK_INPUT,
                        help="mock input JSON text text")
    args = parser.parse_args()

    if args.mode in ("model", "all"):
        measure_model_load()

    if args.mode in ("subprocess", "all"):
        asyncio.run(measure_subprocess(args.concurrency, args.input))


if __name__ == "__main__":
    main()
