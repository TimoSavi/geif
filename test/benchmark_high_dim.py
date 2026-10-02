#!/usr/bin/env python3
"""
benchmark_high_dim.py - High-Dimensional Benchmarking Suite for GEIF.
Compares Bubble Algorithm vs. CEIF across:
  - 4D Concentric Hyperspherical Shell (Hollow Core)
  - 8D Concentric Hyperspherical Shell (Hollow Core)
  - 11D Real-World Tabular: Wine Quality Red (11 features, 6 categories)
  - 42D Complex Tabular: Paddy Dataset (42 features, 36 categories)

Author / Maintainer: Timo Savinen (AI-assisted)
"""

import math
import random
import time
import subprocess
import csv
import statistics
import json
import os
import sys

GEIF_BIN = os.path.abspath("./bin/geif")
WINE_CSV = "/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/winequality-red.csv"
PADDY_CSV = "/home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/paddydataset.csv"
TMP_DIR = "/tmp/geif_benchmark_high_dim"

os.makedirs(TMP_DIR, exist_ok=True)
random.seed(42)

def generate_hypersphere_shell(d, n_samples, r_min, r_max, center=None):
    """Generate n points uniformly distributed on a D-dimensional spherical shell."""
    if center is None:
        center = [0.0] * d
    points = []
    for _ in range(n_samples):
        # Gaussian vector gives uniform random direction
        vec = [random.gauss(0, 1) for _ in range(d)]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-12:
            norm = 1.0
        r = r_min + random.random() * (r_max - r_min)
        pt = [center[j] + (vec[j] / norm) * r for j in range(d)]
        points.append(pt)
    return points

def run_cmd(cmd, input_text=None):
    start = time.perf_counter()
    res = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    elapsed = time.perf_counter() - start
    return res.returncode, res.stdout, res.stderr, elapsed

def parse_scores(output_text, delimiter=','):
    scores = []
    for line in output_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(delimiter)
        # Template %d,%s outputs coords..., score
        try:
            scores.append(float(parts[-1]))
        except ValueError:
            pass
    return scores

def benchmark_shell(d, n_train=2500, r_min=8.0, r_max=12.0, cavity_radius=3.0, outer_radius=25.0):
    print(f"\n=======================================================")
    print(f"BENCHMARK: {d}D Hyperspherical Shell with Hollow Cavity")
    print(f"=======================================================")
    train_pts = generate_hypersphere_shell(d, n_train, r_min, r_max)
    train_file = os.path.join(TMP_DIR, f"train_{d}d.csv")
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for pt in train_pts:
            writer.writerow([f"{x:.6f}" for x in pt])

    # Test points:
    # 1. Inlier points on the shell (r ~ 10.0)
    inlier_pts = generate_hypersphere_shell(d, 200, r_min + 0.5, r_max - 0.5)
    # 2. Hollow cavity points (center r = 0 and r < cavity_radius)
    cavity_pts = [[0.0] * d] + generate_hypersphere_shell(d, 50, 0.1, cavity_radius)
    # 3. Outer space points (r >= outer_radius)
    outer_pts = generate_hypersphere_shell(d, 50, outer_radius, outer_radius * 1.5)

    test_file = os.path.join(TMP_DIR, f"test_{d}d.csv")
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for pt in inlier_pts:
            writer.writerow([f"{x:.6f}" for x in pt])
        for pt in cavity_pts:
            writer.writerow([f"{x:.6f}" for x in pt])
        for pt in outer_pts:
            writer.writerow([f"{x:.6f}" for x in pt])

    results = {}
    for algo in ["bubble", "ceif"]:
        model_file = os.path.join(TMP_DIR, f"model_{algo}_{d}d.json")
        out_file = os.path.join(TMP_DIR, f"scores_{algo}_{d}d.csv")

        # Train
        cmd_train = [GEIF_BIN, "-l", train_file, "-B", algo, "-w", model_file, "-t", "100", "-s", "256"]
        rc, _, err, t_train = run_cmd(cmd_train)
        if rc != 0:
            print(f"Error training {algo}: {err}")
            continue

        model_size = os.path.getsize(model_file)

        # Score
        cmd_score = [GEIF_BIN, "-r", model_file, "-a", test_file, "-p", "%s"]
        rc, out, err, t_score = run_cmd(cmd_score)
        scores = [float(line.strip()) for line in out.strip().splitlines() if line.strip()]

        n_in = len(inlier_pts)
        n_cav = len(cavity_pts)
        n_out = len(outer_pts)

        in_scores = scores[:n_in]
        cav_scores = scores[n_in:n_in + n_cav]
        out_scores = scores[n_in + n_cav:]

        results[algo] = {
            "train_time_ms": t_train * 1000.0,
            "score_time_ms": t_score * 1000.0,
            "score_rate": len(scores) / t_score if t_score > 0 else 0,
            "model_size_kb": model_size / 1024.0,
            "inlier_min": min(in_scores),
            "inlier_mean": statistics.mean(in_scores),
            "inlier_max": max(in_scores),
            "cavity_center_score": cav_scores[0], # exact center (0,0,...,0)
            "cavity_mean": statistics.mean(cav_scores),
            "cavity_min": min(cav_scores),
            "outer_mean": statistics.mean(out_scores),
            "cavity_to_inlier_ratio": statistics.mean(cav_scores) / statistics.mean(in_scores) if statistics.mean(in_scores) > 0 else 0
        }

        print(f"[{algo.upper()}] Train: {t_train*1000:.1f}ms, Score: {t_score*1000:.1f}ms ({results[algo]['score_rate']:.0f} pts/sec)")
        print(f"  Inlier shell mean: {results[algo]['inlier_mean']:.4f} (range: {results[algo]['inlier_min']:.4f} - {results[algo]['inlier_max']:.4f})")
        print(f"  Cavity center (0): {results[algo]['cavity_center_score']:.4f}")
        print(f"  Cavity region mean: {results[algo]['cavity_mean']:.4f} (contrast ratio vs inlier: {results[algo]['cavity_to_inlier_ratio']:.2f}x)")
        print(f"  Outer space mean:  {results[algo]['outer_mean']:.4f}")

    return results

def benchmark_tabular(name, csv_path, delimiter, cat_range, ignore_range=None, skip_header=True):
    print(f"\n=======================================================")
    print(f"BENCHMARK: {name}")
    print(f"=======================================================")
    results = {}
    for algo in ["bubble", "ceif"]:
        model_file = os.path.join(TMP_DIR, f"model_{name}_{algo}.json")
        out_file = os.path.join(TMP_DIR, f"scores_{name}_{algo}.csv")

        train_cmd = [GEIF_BIN, "-l", csv_path, "-f", delimiter, "-C", cat_range, "-B", algo, "-w", model_file, "-t", "50", "-s", "128"]
        if ignore_range:
            train_cmd.extend(["-I", ignore_range])
        if skip_header:
            train_cmd.append("-H")

        rc, out, err, t_train = run_cmd(train_cmd)
        if rc != 0:
            print(f"Error training {name} with {algo}: {err}")
            continue

        model_size = os.path.getsize(model_file)

        # Query model diagnostics
        _, q_out, _, _ = run_cmd([GEIF_BIN, "-r", model_file, "-q"])
        dims = 0
        subforests = 0
        for line in q_out.splitlines():
            if "Dimensions:" in line:
                dims = int(line.split()[-1])
            if "Sub-forests:" in line:
                subforests = int(line.split()[-1])

        # Score the whole dataset
        score_cmd = [GEIF_BIN, "-r", model_file, "-a", csv_path, "-f", delimiter, "-p", "%s"]
        if skip_header:
            score_cmd.append("-H")

        rc, s_out, err, t_score = run_cmd(score_cmd)
        scores = [float(l.strip()) for l in s_out.strip().splitlines() if l.strip()]

        sorted_s = sorted(scores)
        n = len(scores)
        p50 = sorted_s[n // 2]
        p95 = sorted_s[int(n * 0.95)]
        p99 = sorted_s[int(n * 0.99)]

        results[algo] = {
            "dimensions": dims,
            "subforests": subforests,
            "rows": n,
            "train_time_ms": t_train * 1000.0,
            "score_time_ms": t_score * 1000.0,
            "score_rate": n / t_score if t_score > 0 else 0,
            "model_size_kb": model_size / 1024.0,
            "score_min": sorted_s[0],
            "score_mean": statistics.mean(scores),
            "score_p50": p50,
            "score_p95": p95,
            "score_p99": p99,
            "score_max": sorted_s[-1],
            "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0
        }

        print(f"[{algo.upper()}] Dims: {dims}, Sub-forests: {subforests}, Rows: {n}")
        print(f"  Train: {t_train*1000:.1f}ms, Score: {t_score*1000:.1f}ms ({results[algo]['score_rate']:.0f} rows/sec)")
        print(f"  Model Size: {results[algo]['model_size_kb']:.1f} KB")
        print(f"  Score Min: {results[algo]['score_min']:.4f}, Mean: {results[algo]['score_mean']:.4f}, Median: {p50:.4f}")
        print(f"  Percentiles: 95th={p95:.4f}, 99th={p99:.4f}, Max={sorted_s[-1]:.4f}")

    return results

if __name__ == "__main__":
    report_data = {}
    report_data["shell_4d"] = benchmark_shell(4, n_train=3000, r_min=8.0, r_max=12.0, cavity_radius=3.0, outer_radius=25.0)
    report_data["shell_8d"] = benchmark_shell(8, n_train=4000, r_min=10.0, r_max=15.0, cavity_radius=4.0, outer_radius=30.0)
    report_data["wine_11d"] = benchmark_tabular("WineQuality-Red_11D", WINE_CSV, delimiter=';', cat_range="12")
    report_data["paddy_42d"] = benchmark_tabular("PaddyDataset_42D", PADDY_CSV, delimiter=',', cat_range="2-4")

    # Output JSON summary for report authoring
    summary_path = os.path.join(TMP_DIR, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved raw benchmark data to {summary_path}")
