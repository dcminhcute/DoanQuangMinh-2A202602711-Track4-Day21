"""So khớp cluster với GT box (KITTI label_2), tính precision/recall/F1.

Chạy:
    python -m src.obstacle_evaluate --data-root data/kitti_mini
    python -m src.obstacle_evaluate --data-root data/kitti_mini --distance-threshold 0.3
    python -m src.obstacle_evaluate --help
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from starter.datasets import load_frame, list_frames
from starter.kitti_io import KittiObject
from src.obstacle_detection import detect_obstacles, ObstacleCluster


def box3d_to_corners(obj: KittiObject, calib) -> np.ndarray:
    """Chuyển GT box 3D từ camera frame sang velodyne frame, trả về (8, 3)."""
    h, w, l = obj.dimensions
    # 8 góc trong camera frame
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners_cam = (R @ np.vstack([x, y, z])).T + obj.location  # (8, 3) camera frame

    # Chuyển về velodyne frame: T_velo_cam = inv(T_cam_velo)
    T = calib.T_cam_velo  # 4x4
    T_inv = np.linalg.inv(T)
    ones = np.ones((8, 1))
    corners_homo = np.hstack([corners_cam, ones])  # (8, 4)
    corners_velo = (T_inv @ corners_homo.T).T[:, :3]
    return corners_velo


def gt_box_bounds_velo(obj: KittiObject, calib) -> tuple[np.ndarray, np.ndarray]:
    """Min/max bounds (3,) của GT box trong velodyne frame."""
    corners = box3d_to_corners(obj, calib)
    return corners.min(axis=0), corners.max(axis=0)


def cluster_overlaps_gt(cluster: ObstacleCluster,
                        gt_min: np.ndarray, gt_max: np.ndarray) -> bool:
    """Kiểm tra AABB overlap giữa cluster và GT box."""
    return bool(
        np.all(cluster.bbox_min <= gt_max) and np.all(cluster.bbox_max >= gt_min)
    )


def evaluate_frame(data_root: str, frame_id: str, config: dict) -> dict:
    """Đánh giá 1 frame: so khớp clusters với GT boxes."""
    fr = load_frame(data_root, frame_id)
    points = fr["points"]
    calib = fr["calib"]
    labels = [obj for obj in fr["labels"] if obj.type not in ("DontCare",)]

    result = detect_obstacles(points, **config)
    clusters = result.clusters

    # Với mỗi GT, tìm cluster overlap
    gt_matched = [False] * len(labels)
    cluster_matched = [False] * len(clusters)

    for gi, obj in enumerate(labels):
        gt_min, gt_max = gt_box_bounds_velo(obj, calib)
        for ci, cl in enumerate(clusters):
            if cluster_overlaps_gt(cl, gt_min, gt_max):
                gt_matched[gi] = True
                cluster_matched[ci] = True

    tp = sum(gt_matched)  # GT boxes được detect
    fn = len(labels) - tp  # GT boxes bị bỏ sót
    fp = sum(not m for m in cluster_matched)  # clusters không khớp GT nào

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "frame_id": frame_id,
        "n_gt": len(labels),
        "n_clusters": len(clusters),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        **{f"config_{k}": v for k, v in config.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="So khớp cluster với GT box, tính precision/recall/F1"
    )
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frames", nargs="+", default=None,
                    help="frame ids (mặc định: tất cả)")
    ap.add_argument("--voxel-size", type=float, default=0.1)
    ap.add_argument("--distance-threshold", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--min-points", type=int, default=10)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--out", default="results/obstacle_evaluation.csv")
    args = ap.parse_args()

    frames = args.frames or list_frames(args.data_root)
    config = {
        "voxel_size": args.voxel_size,
        "distance_threshold": args.distance_threshold,
        "eps": args.eps,
        "min_points": args.min_points,
        "max_range": args.max_range,
    }

    rows = []
    for fid in frames:
        row = evaluate_frame(args.data_root, fid, config)
        rows.append(row)
        print(f"  {fid}: GT={row['n_gt']} clusters={row['n_clusters']} "
              f"P={row['precision']:.2f} R={row['recall']:.2f} F1={row['f1']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")

    # Summary
    avg_p = np.mean([r["precision"] for r in rows])
    avg_r = np.mean([r["recall"] for r in rows])
    avg_f1 = np.mean([r["f1"] for r in rows])
    print(f"Average: Precision={avg_p:.3f} Recall={avg_r:.3f} F1={avg_f1:.3f}")


if __name__ == "__main__":
    main()
