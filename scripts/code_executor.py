"""C++ compile-and-run sandbox for competitive programming reward computation.

Used by the CodeContests training pipeline to score model completions against
problem test cases. The executor is JAX-free and runs on the CPU side after
detokenization, parallelized across completions with a process pool.

Pipeline per completion:
    extract_cpp(text) -> source       # strip markdown fences
    execute_cpp(source, tests) -> ExecutionResult
        compile with g++ -O2 -std=c++17
        for each (stdin, expected_stdout):
            run binary under setrlimit + timeout
            judge stdout vs expected
        return per-test verdicts and pass_fraction

Sandboxing is best-effort (subprocess + setrlimit + process-group kill on
timeout). It assumes a non-adversarial model — fine for RL on a trusted host,
not a substitute for a real sandbox if you ever expose this to untrusted code.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

Status = Literal["AC", "WA", "TLE", "MLE", "RE", "CE"]

# Truncate captured stdout/stderr to keep ExecutionResult small in memory and
# in JSON dumps. Test outputs in CodeContests are typically tiny; long output
# almost always means a runaway loop printing garbage.
_OUTPUT_TRUNCATE_BYTES = 4096

_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|C\+\+)\s*\n(.*?)```", re.DOTALL)


@dataclass
class TestResult:
    status: Status
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0


@dataclass
class ExecutionResult:
    compiled: bool
    compile_error: Optional[str] = None
    test_results: list[TestResult] = field(default_factory=list)
    pass_fraction: float = 0.0


# ── Code extraction ───────────────────────────────────────────────────────────


def extract_cpp(text: str) -> Optional[str]:
    """Extract C++ source from a model completion.

    Returns the contents of the first ```cpp / ```c++ fenced block, or None
    if no such fence is found.
    """
    matches = _CPP_FENCE_RE.findall(text)
    return matches[-1] if matches else None


# ── Sandbox setup (Linux; runs in the child process) ──────────────────────────


def _apply_rlimits(mem_limit_mb: int, cpu_limit_s: int) -> None:
    """Apply resource limits in the forked child before exec.

    - RLIMIT_AS: virtual memory cap (catches malloc bombs, large arrays).
    - RLIMIT_CPU: CPU-second cap (backstop for the wall-clock timeout).
    - RLIMIT_FSIZE: prevents writing huge files to /tmp.
    - RLIMIT_NPROC: tight cap (4) so multiprocessing.Pool / fork-based
      parallelism fails immediately. Headroom above 1 is for transient
      threads from libc / Python runtime; not enough for any real pool.
    - setpgrp: detach into our own process group so we can kill descendants.
    - sched_setaffinity: pin to a single core so multithreaded solutions can't
      mask a slow algorithm by burning multiple cores. Pick one from the
      parent's existing affinity mask, spread by PID so concurrent runs don't
      all serialize onto core 0.
    """
    mem_bytes = mem_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (4, 4))
    os.setpgrp()

    if hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            os.sched_setaffinity(0, {allowed[os.getpid() % len(allowed)]})


# ── Compile + run primitives ──────────────────────────────────────────────────


def _truncate(b: bytes) -> str:
    if len(b) > _OUTPUT_TRUNCATE_BYTES:
        b = b[:_OUTPUT_TRUNCATE_BYTES] + b"\n...[truncated]"
    return b.decode("utf-8", errors="replace")


def _compile(source: str, workdir: Path, compile_timeout_s: float) -> tuple[Optional[Path], Optional[str]]:
    """Compile a C++ source string. Returns (binary_path, error_message)."""
    src_path = workdir / "sol.cpp"
    bin_path = workdir / "sol"
    src_path.write_text(source)

    try:
        proc = subprocess.run(
            ["g++", "-O2", "-std=c++17", "-pipe", "-w", "-o", str(bin_path), str(src_path)],
            capture_output=True,
            timeout=compile_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "compile timeout"
    except FileNotFoundError:
        return None, "g++ not found on PATH"

    if proc.returncode != 0:
        return None, _truncate(proc.stderr)
    return bin_path, None


def _run_one(
    bin_path: Path,
    stdin_data: str,
    time_limit_s: float,
    mem_limit_mb: int,
) -> TestResult:
    """Run the compiled binary on a single test input."""
    cpu_limit_s = max(1, int(time_limit_s) + 1)
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [str(bin_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=lambda: _apply_rlimits(mem_limit_mb, cpu_limit_s),
        )
    except OSError as e:
        return TestResult(status="RE", stderr=f"spawn failed: {e}", elapsed_s=0.0)

    try:
        stdout_b, stderr_b = proc.communicate(
            input=stdin_data.encode("utf-8"), timeout=time_limit_s
        )
    except subprocess.TimeoutExpired:
        # Kill the whole process group so any forked descendants die too.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return TestResult(status="TLE", elapsed_s=time.time() - t0)

    elapsed = time.time() - t0
    stdout = _truncate(stdout_b)
    stderr = _truncate(stderr_b)

    if proc.returncode != 0:
        # SIGKILL from RLIMIT_AS shows up as -9 → classify as MLE if stderr is
        # empty (kernel kill leaves no message). Best-effort heuristic.
        if proc.returncode == -9 and not stderr_b:
            return TestResult(status="MLE", stdout=stdout, stderr=stderr, elapsed_s=elapsed)
        return TestResult(status="RE", stdout=stdout, stderr=stderr, elapsed_s=elapsed)

    return TestResult(status="AC", stdout=stdout, stderr=stderr, elapsed_s=elapsed)


# ── Output judging ────────────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Strip trailing whitespace per line and trailing blank lines."""
    lines = [line.rstrip() for line in s.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _judge(actual_stdout: str, expected_stdout: str) -> bool:
    return _normalize(actual_stdout) == _normalize(expected_stdout)


# ── Public entry points ───────────────────────────────────────────────────────


def execute_cpp(
    source: str,
    tests: list[tuple[str, str]],
    time_limit_s: float = 5.0,
    mem_limit_mb: int = 512,
    compile_timeout_s: float = 10.0,
) -> ExecutionResult:
    """Compile a C++ source and run it against a list of (stdin, expected_stdout) tests.

    On compile failure: returns compiled=False, no test results, pass_fraction=0.
    On compile success: runs every test and judges output. Test status is AC if
    output matches expected; otherwise WA / TLE / MLE / RE.
    """
    if not tests:
        return ExecutionResult(compiled=False, compile_error="no tests provided")

    with tempfile.TemporaryDirectory(prefix="cpp_exec_") as td:
        workdir = Path(td)
        bin_path, compile_err = _compile(source, workdir, compile_timeout_s)
        if bin_path is None:
            return ExecutionResult(compiled=False, compile_error=compile_err)

        results: list[TestResult] = []
        passed = 0
        for stdin_data, expected in tests:
            tr = _run_one(bin_path, stdin_data, time_limit_s, mem_limit_mb)
            if tr.status == "AC":
                if _judge(tr.stdout, expected):
                    passed += 1
                else:
                    tr = TestResult(
                        status="WA",
                        stdout=tr.stdout,
                        stderr=tr.stderr,
                        elapsed_s=tr.elapsed_s,
                    )
            results.append(tr)

        return ExecutionResult(
            compiled=True,
            compile_error=None,
            test_results=results,
            pass_fraction=passed / len(tests),
        )


def _execute_one_kwargs(args: tuple) -> ExecutionResult:
    """Pool worker entry point — unpacks args and calls execute_cpp."""
    source, tests, time_limit_s, mem_limit_mb, compile_timeout_s = args
    return execute_cpp(
        source,
        tests,
        time_limit_s=time_limit_s,
        mem_limit_mb=mem_limit_mb,
        compile_timeout_s=compile_timeout_s,
    )


def execute_cpp_batch(
    items: list[tuple[str, list[tuple[str, str]]]],
    num_workers: Optional[int] = None,
    time_limit_s: float = 5.0,
    mem_limit_mb: int = 512,
    compile_timeout_s: float = 10.0,
) -> list[ExecutionResult]:
    """Run execute_cpp over many (source, tests) pairs in parallel.

    Each item gets its own worker process; the pool size defaults to
    cpu_count - 2 (leave headroom for the main JAX/serving process).

    A None source (e.g. extract_cpp returned None) is reported as a
    compile failure with error "no code extracted" — caller doesn't have
    to filter beforehand.
    """
    if not items:
        return []

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 2)
    num_workers = min(num_workers, len(items))

    # Items with no extracted source short-circuit without spawning a worker.
    payload = []
    none_indices = []
    for i, (src, tests) in enumerate(items):
        if src is None:
            none_indices.append(i)
            payload.append(None)
        else:
            payload.append((src, tests, time_limit_s, mem_limit_mb, compile_timeout_s))

    real_indices = [i for i, p in enumerate(payload) if p is not None]
    real_payload = [payload[i] for i in real_indices]

    results: list[Optional[ExecutionResult]] = [None] * len(items)

    if real_payload:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_workers) as pool:
            real_results = pool.map(_execute_one_kwargs, real_payload)
        for idx, res in zip(real_indices, real_results):
            results[idx] = res

    for i in none_indices:
        results[i] = ExecutionResult(compiled=False, compile_error="no code extracted")

    return [r for r in results if r is not None]
