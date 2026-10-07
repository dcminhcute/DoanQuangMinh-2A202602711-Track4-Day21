"""Tạo ảnh demo, biểu đồ sweep, và failure case cho topic D.

Chạy:
    python -m src.obstacle_visualize --data-root data/kitti_mini --frame 000011
    python -m src.obstacle_visualize --plot-sweeps --csv-dir results
    python -m src.obstacle_visualize --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import load_points
from src.obstacle_detection import detect_obstacles, DetectionResult


# ---------------------------------------------------------------------------
# BEV (Bird's Eye View) plotting
# ---------------------------------------------------------------------------
def plot_bev(result: DetectionResult, title: str = "", ax: plt.Axes | None = None,
             xlim: tuple = (-40, 40), ylim: tuple = (-5, 80)) -> plt.Axes:
    """Vẽ BEV: ground = xám, clusters = màu, noise = đen nhạt."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Ground
    g = result.ground_points
    if len(g):
        ax.scatter(g[:, 0], g[:, 1], s=0.2, c="lightgray", alpha=0.3, label="ground")

    # Non-ground by cluster
    ng = result.non_ground_points
    labels = result.labels
    if len(ng):
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(ng[noise_mask, 0], ng[noise_mask, 1], s=0.3, c="gray",
                       alpha=0.2, label="noise")
        cluster_mask = ~noise_mask
        if cluster_mask.any():
            ax.scatter(ng[cluster_mask, 0], ng[cluster_mask, 1], s=1,
                       c=labels[cluster_mask], cmap="tab20", alpha=0.7)

    # Bounding boxes
    for c in result.clusters:
        rect = plt.Rectangle(
            (c.bbox_min[0], c.bbox_min[1]),
            c.size[0], c.size[1],
            linewidth=1, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(c.center[0], c.center[1], f"{c.distance:.0f}m",
                fontsize=5, color="red", ha="center")

    # Ego marker
    ax.plot(0, 0, "^", color="blue", markersize=8, label="ego")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(title or f"{len(result.clusters)} clusters")
    ax.legend(loc="upper right", fontsize=6)
    return ax


def plot_pipeline_steps(result: DetectionResult, frame_id: str,
                        save_path: str | Path) -> None:
    """4-panel figure: raw → downsample → ground removal → clusters."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # 1. Raw
    raw = result.raw_points
    axes[0].scatter(raw[:, 0], raw[:, 1], s=0.1, c="black", alpha=0.3)
    axes[0].set_title(f"1. Raw ({len(raw)} pts)")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-40, 40); axes[0].set_ylim(-5, 80)

    # 2. Downsampled
    down = result.downsampled_points
    axes[1].scatter(down[:, 0], down[:, 1], s=0.3, c="black", alpha=0.3)
    axes[1].set_title(f"2. Downsample ({len(down)} pts, "
                      f"voxel={result.config['voxel_size']}m)")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-40, 40); axes[1].set_ylim(-5, 80)

    # 3. Ground removal
    g = result.ground_points
    ng = result.non_ground_points
    if len(g):
        axes[2].scatter(g[:, 0], g[:, 1], s=0.2, c="green", alpha=0.3, label="ground")
    if len(ng):
        axes[2].scatter(ng[:, 0], ng[:, 1], s=0.3, c="red", alpha=0.4, label="non-ground")
    axes[2].set_title(f"3. Ground removal\n"
                      f"(ground={len(g)}, obstacles={len(ng)}, "
                      f"thresh={result.config['distance_threshold']}m)")
    axes[2].set_aspect("equal")
    axes[2].set_xlim(-40, 40); axes[2].set_ylim(-5, 80)
    axes[2].legend(fontsize=6)

    # 4. Clusters
    plot_bev(result, title=f"4. Clusters ({len(result.clusters)} obstacles, "
             f"eps={result.config['eps']}m)", ax=axes[3])

    for ax in axes:
        ax.plot(0, 0, "^", color="blue", markersize=6)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.suptitle(f"Obstacle Detection Pipeline — Frame {frame_id}", fontsize=14, y=1.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {save_path}")


def plot_sweep_results(csv_path: str | Path, param: str, save_path: str | Path) -> None:
    """Vẽ biểu đồ kết quả sweep."""
    df = pd.read_csv(csv_path)
    # Trung bình qua các frame
    agg = df.groupby(param).agg({
        "n_clusters": "mean",
        "ground_ratio": "mean",
        "nearest_obstacle_m": "mean",
        "largest_cluster_pts": "mean",
        "latency_p50_ms": "mean",
        "latency_p95_ms": "mean",
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Số clusters
    axes[0, 0].bar(range(len(agg)), agg["n_clusters"], color="steelblue")
    axes[0, 0].set_xticks(range(len(agg)))
    axes[0, 0].set_xticklabels([f"{v}" for v in agg[param]])
    axes[0, 0].set_xlabel(param)
    axes[0, 0].set_ylabel("Avg clusters")
    axes[0, 0].set_title("Number of Clusters")

    # Ground ratio
    axes[0, 1].plot(agg[param], agg["ground_ratio"] * 100, "o-", color="green")
    axes[0, 1].set_xlabel(param)
    axes[0, 1].set_ylabel("Ground ratio (%)")
    axes[0, 1].set_title("Ground Points Ratio")

    # Nearest obstacle
    axes[1, 0].plot(agg[param], agg["nearest_obstacle_m"], "s-", color="red")
    axes[1, 0].set_xlabel(param)
    axes[1, 0].set_ylabel("Distance (m)")
    axes[1, 0].set_title("Nearest Obstacle Distance")

    # Latency
    axes[1, 1].plot(agg[param], agg["latency_p50_ms"], "o-", label="p50", color="orange")
    axes[1, 1].plot(agg[param], agg["latency_p95_ms"], "s--", label="p95", color="red")
    axes[1, 1].set_xlabel(param)
    axes[1, 1].set_ylabel("Latency (ms)")
    axes[1, 1].set_title("Pipeline Latency")
    axes[1, 1].legend()

    fig.suptitle(f"Sweep: {param}", fontsize=14)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {save_path}")


def plot_failure_case(result: DetectionResult, result_ref: DetectionResult,
                      frame_id: str, failure_desc: str,
                      save_path: str | Path) -> None:
    """So sánh 2 kết quả (tốt vs xấu) để minh hoạ failure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_bev(result_ref, title=f"Reference config\n"
             f"(thresh={result_ref.config['distance_threshold']}m, "
             f"eps={result_ref.config['eps']}m)\n"
             f"{len(result_ref.clusters)} clusters", ax=axes[0])
    plot_bev(result, title=f"Failure config\n"
             f"(thresh={result.config['distance_threshold']}m, "
             f"eps={result.config['eps']}m)\n"
             f"{len(result.clusters)} clusters", ax=axes[1])

    fig.suptitle(f"Failure Case — Frame {frame_id}\n{failure_desc}", fontsize=12, y=1.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo ảnh demo và biểu đồ cho topic D")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--plot-sweeps", action="store_true", help="vẽ biểu đồ từ CSV sweep")
    ap.add_argument("--csv-dir", default="results")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.plot_sweeps:
        for param in ["voxel_size", "distance_threshold", "eps"]:
            csv_path = Path(args.csv_dir) / f"obstacle_sweep_{param}.csv"
            if csv_path.exists():
                plot_sweep_results(csv_path, param,
                                   out / f"sweep_{param}.png")
    else:
        # Demo pipeline
        points = load_points(args.data_root, args.frame)
        result = detect_obstacles(points, voxel_size=0.1, distance_threshold=0.2,
                                  eps=0.5, min_points=10)
        plot_pipeline_steps(result, args.frame,
                            out / f"demo_pipeline_{args.frame}.png")

        # Failure case 1: threshold quá lớn → mất vật thấp
        result_bad = detect_obstacles(points, voxel_size=0.1, distance_threshold=0.5,
                                      eps=0.5, min_points=10)
        plot_failure_case(
            result_bad, result, args.frame,
            "distance_threshold=0.5m: vật thấp sát đất bị RANSAC xếp vào mặt đất",
            out / f"fail_01_high_threshold_{args.frame}.png"
        )

        # Failure case 2: eps quá lớn → cluster bị gộp
        result_merge = detect_obstacles(points, voxel_size=0.1, distance_threshold=0.2,
                                        eps=2.0, min_points=10)
        plot_failure_case(
            result_merge, result, args.frame,
            "eps=2.0m: DBSCAN gộp nhiều vật gần nhau thành 1 cluster lớn",
            out / f"fail_02_merged_clusters_{args.frame}.png"
        )


if __name__ == "__main__":
    main()
