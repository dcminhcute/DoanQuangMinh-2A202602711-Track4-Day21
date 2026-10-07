"""Sweep tham số pipeline phát hiện vật cản, đo metrics + latency p50/p95.

Chạy:
    python -m src.obstacle_sweep --data-root data/kitti_mini --sweep all
    python -m src.obstacle_sweep --data-root data/kitti_mini --sweep voxel_size
    python -m src.obstacle_sweep --help
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from starter.datasets import load_points, list_frames
from src.obstacle_detection import detect_obstacles


def run_sweep(
    data_root: str,
    frame_ids: list[str],
    sweep_param: str,
    sweep_values: list[float],
    base_config: dict,
    n_repeats: int = 21,
) -> list[dict]:
    """Chạy pipeline nhiều lần, sweep 1 tham số, đo metrics + latency."""
    rows = []
    for val in sweep_values:
        config = {**base_config, sweep_param: val}
        for fid in frame_ids:
            points = load_points(data_root, fid)
            # Warm-up run (bỏ)
            _ = detect_obstacles(points, **config)
            # Đo latency
            latencies = []
            result = None
            for i in range(n_repeats):
                result = detect_obstacles(points, **config)
                latencies.append(result.t_total_ms)
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            row = {
                "frame_id": fid,
                sweep_param: val,
                "n_raw": len(result.raw_points),
                "n_downsampled": len(result.downsampled_points),
                "n_ground": len(result.ground_points),
                "n_non_ground": len(result.non_ground_points),
                "ground_ratio": round(len(result.ground_points) / max(1, len(result.downsampled_points)), 4),
                "n_clusters": len(result.clusters),
                "n_noise": int((result.labels == -1).sum()) if len(result.labels) > 0 else 0,
                "nearest_obstacle_m": round(min((c.distance for c in result.clusters), default=float("nan")), 2),
                "largest_cluster_pts": max((c.n_points for c in result.clusters), default=0),
                "mean_cluster_height": round(float(np.mean([c.height for c in result.clusters])), 3) if result.clusters else float("nan"),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
            }
            rows.append(row)
            print(f"  {sweep_param}={val}, frame={fid}: "
                  f"clusters={row['n_clusters']}, nearest={row['nearest_obstacle_m']}m, "
                  f"latency_p50={row['latency_p50_ms']:.1f}ms")
    return rows


def save_csv(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {p}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep tham số pipeline phát hiện vật cản, đo metrics + latency p50/p95"
    )
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="+", default=["000011", "000001", "000008"],
                    help="danh sách frame id")
    ap.add_argument("--sweep", choices=["voxel_size", "distance_threshold", "eps", "all"],
                    default="all", help="tham số cần sweep")
    ap.add_argument("--n-repeats", type=int, default=21, help="số lần chạy lại để đo latency")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    base = {"voxel_size": 0.1, "distance_threshold": 0.2, "eps": 0.5,
            "min_points": 10, "max_range": 80.0}

    sweeps = {}
    if args.sweep in ("voxel_size", "all"):
        sweeps["voxel_size"] = [0.05, 0.1, 0.2, 0.3, 0.5]
    if args.sweep in ("distance_threshold", "all"):
        sweeps["distance_threshold"] = [0.1, 0.15, 0.2, 0.3, 0.5]
    if args.sweep in ("eps", "all"):
        sweeps["eps"] = [0.3, 0.5, 0.8, 1.0, 1.5]

    for param, values in sweeps.items():
        print(f"\n=== Sweep {param}: {values} ===")
        rows = run_sweep(args.data_root, args.frames, param, values, base, args.n_repeats)
        save_csv(rows, Path(args.out_dir) / f"obstacle_sweep_{param}.csv")


if __name__ == "__main__":
    main()
