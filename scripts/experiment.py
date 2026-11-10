import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_config(path):
    """Read the simple key=value configuration used by the repository."""
    values = {}

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()

        if not line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def write_config(values, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items())
    )


def run_program(executable, config, threads, log_prefix):
    env = os.environ.copy()

    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"

    start = time.perf_counter()

    result = subprocess.run(
        [str(ROOT / executable), str(config)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    seconds = time.perf_counter() - start

    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.txt").write_text(result.stdout)
    log_prefix.with_suffix(".stderr.txt").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"{executable} failed: exit code {result.returncode}. "
            f"See {log_prefix}.stderr.txt"
        )

    return seconds


def prepare_case(base, directory, output_enabled):
    values = dict(base)

    # Use simple relative paths understood by the original parameter parser.
    relative_prefix = (
        directory.relative_to(ROOT) / "traffic"
    ).as_posix()

    values["outputprefix"] = relative_prefix

    if not output_enabled:
        values["per"] = "0"

    config = directory / "config.ini"
    write_config(values, config)

    return config


def compare_outputs(reference_dir, candidate_dir):
    for field in ("dens", "velo", "time"):
        name = f"traffic-{field}.npy"

        reference = np.load(reference_dir / name, allow_pickle=False)
        candidate = np.load(candidate_dir / name, allow_pickle=False)

        if reference.shape != candidate.shape:
            raise AssertionError(
                f"{name}: shape differs: "
                f"{reference.shape} != {candidate.shape}"
            )

        if reference.dtype != candidate.dtype:
            raise AssertionError(
                f"{name}: dtype differs: "
                f"{reference.dtype} != {candidate.dtype}"
            )

        if not np.array_equal(reference, candidate):
            index = tuple(
                np.argwhere(reference != candidate)[0]
            )

            raise AssertionError(
                f"{name}: first mismatch at {index}: "
                f"{reference[index]} != {candidate[index]}"
            )


def check(args):
    base = read_config(args.config)

    if int(base.get("per", "0")) <= 0:
        raise ValueError("Correctness checking requires per > 0.")

    directory = RESULTS / "correctness"
    reference_dir = directory / "serial"

    config = prepare_case(base, reference_dir, output_enabled=True)

    run_program(
        "traffic_serial",
        config,
        1,
        reference_dir / "run",
    )

    for threads in args.threads:
        candidate_dir = directory / f"omp-{threads}"

        config = prepare_case(
            base, candidate_dir, output_enabled=True
        )

        run_program(
            "traffic_omp",
            config,
            threads,
            candidate_dir / "run",
        )

        compare_outputs(reference_dir, candidate_dir)

        print(f"PASS: serial == OpenMP with {threads} threads")


def benchmark(args):
    base = read_config(args.config)

    directory = RESULTS / "benchmarks"
    directory.mkdir(parents=True, exist_ok=True)

    # Disable output while preserving other supplied parameters.
    config = prepare_case(base, directory, output_enabled=False)

    metadata = {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "compiler": subprocess.check_output(
            ["g++", "--version"], text=True
        ),
        "cpu_info": subprocess.check_output(
            ["lscpu"], text=True
        ),
        "config": config.read_text(),
        "makefile": (ROOT / "Makefile").read_text(),
        "timing_scope": (
            "Whole executable wall time, including startup, "
            "initialization and simulation; per=0."
        ),
        "openmp_environment": {
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        },
    }

    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    cases = [("serial", 1)]
    cases.extend(("omp", n) for n in args.threads)

    # Warm up each executable/thread-count configuration once.
    for implementation, threads in cases:
        executable = f"traffic_{implementation}"

        run_program(
            executable,
            config,
            threads,
            directory / f"warmup-{implementation}-{threads}",
        )

    measurements = []

    # Shuffle configurations within each repetition to reduce order bias.
    order_rng = random.Random(13)

    output = directory / "timings.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "implementation",
                "threads",
                "repetition",
                "seconds",
            ],
        )
        writer.writeheader()

        for repetition in range(1, args.repeats + 1):
            order = cases.copy()
            order_rng.shuffle(order)

            for implementation, threads in order:
                executable = f"traffic_{implementation}"

                seconds = run_program(
                    executable,
                    config,
                    threads,
                    directory /
                    f"run-{implementation}-{threads}-{repetition}",
                )

                row = {
                    "implementation": implementation,
                    "threads": threads,
                    "repetition": repetition,
                    "seconds": seconds,
                }

                writer.writerow(row)
                file.flush()
                measurements.append(row)

                print(
                    f"{implementation:6s} "
                    f"threads={threads:2d} "
                    f"repeat={repetition} "
                    f"time={seconds:.6f}s",
                    flush=True,
                )

    print(f"Saved measurements: {output}")


def plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = RESULTS / "benchmarks"

    with (directory / "timings.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))

    groups = {}

    for row in rows:
        key = (row["implementation"], int(row["threads"]))
        groups.setdefault(key, []).append(float(row["seconds"]))

    serial_time = statistics.median(groups[("serial", 1)])

    threads = sorted(
        n for implementation, n in groups
        if implementation == "omp"
    )

    runtimes = [
        statistics.median(groups[("omp", n)])
        for n in threads
    ]

    speedups = [serial_time / value for value in runtimes]
    efficiencies = [
        speedup / n for speedup, n in zip(speedups, threads)
    ]

    summary = []

    for n, runtime, speedup, efficiency in zip(
        threads, runtimes, speedups, efficiencies
    ):
        values = groups[("omp", n)]

        # Effective fraction relative to the original serial baseline.
        f_effective = (
            (1.0 / speedup - 1.0 / n) / (1.0 - 1.0 / n)
            if n > 1 else None
        )

        summary.append({
            "threads": n,
            "serial_median_seconds": serial_time,
            "omp_median_seconds": runtime,
            "omp_min_seconds": min(values),
            "omp_max_seconds": max(values),
            "speedup": speedup,
            "efficiency": efficiency,
            "effective_serial_fraction": f_effective,
        })

    with (directory / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    lower = [
        median - min(groups[("omp", n)])
        for n, median in zip(threads, runtimes)
    ]
    upper = [
        max(groups[("omp", n)]) - median
        for n, median in zip(threads, runtimes)
    ]

    axes[0].errorbar(
        threads,
        runtimes,
        yerr=[lower, upper],
        marker="o",
        capsize=4,
        label="OpenMP: median, min/max",
    )
    axes[0].axhline(
        serial_time,
        linestyle="--",
        color="black",
        label="Serial median",
    )
    axes[0].set_ylabel("Wall time (seconds)")
    axes[0].legend()

    axes[1].plot(threads, speedups, "o-", label="Measured")
    axes[1].plot(threads, threads, "--", label="Ideal S(P)=P")
    axes[1].axhline(1, color="gray", linewidth=1)
    axes[1].set_ylabel("Speedup vs original serial")
    axes[1].legend()

    axes[2].plot(
        threads,
        [100 * value for value in efficiencies],
        "o-",
    )
    axes[2].axhline(100, linestyle="--", color="gray")
    axes[2].set_ylabel("Parallel efficiency (%)")

    for axis in axes:
        axis.set_xlabel("Threads")
        axis.set_xticks(threads)
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=180)

    print(f"Serial median: {serial_time:.6f}s")
    print("Threads | OpenMP median | Speedup | Efficiency")

    for row in summary:
        print(
            f"{row['threads']:7d} | "
            f"{row['omp_median_seconds']:13.6f} | "
            f"{row['speedup']:7.3f} | "
            f"{100 * row['efficiency']:9.2f}%"
        )

    print(f"Saved figure: {directory / 'comparison.png'}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("check", "benchmark"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument(
            "--threads", nargs="+", type=int, default=[1, 2, 4]
        )

        if name == "benchmark":
            subparser.add_argument("--repeats", type=int, default=5)

    subparsers.add_parser("plot")

    args = parser.parse_args()

    if hasattr(args, "threads"):
        if not args.threads or any(n <= 0 for n in args.threads):
            parser.error("Thread counts must be positive.")

        args.threads = sorted(set(args.threads))

    if hasattr(args, "repeats") and args.repeats <= 0:
        parser.error("repeats must be positive.")

    if hasattr(args, "config"):
        args.config = args.config.resolve()

    {"check": check, "benchmark": benchmark, "plot": plot}[
        args.command
    ](args)


if __name__ == "__main__":
    main()