"""Pipeline phát hiện vật cản cho robot/drone — không dùng deep learning.

Pipeline 4 bước:
    1. Voxel downsample
    2. Ground removal (RANSAC plane fitting)
    3. Clustering (DBSCAN)
    4. Bounding box cho mỗi cluster

Chạy demo:
    python -m src.obstacle_detection --data-root data/kitti_mini --frame 000011
    python -m src.obstacle_detection --data-root data/kitti_mini --frame 000011 --voxel-size 0.2
    python -m src.obstacle_detection --help
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Cần cài open3d: pip install open3d>=0.18")

from starter.datasets import load_points, list_frames


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ObstacleCluster:
    """Một cụm vật cản phát hiện được."""
    cluster_id: int
    points: np.ndarray          # (M, 3)
    center: np.ndarray          # (3,) trung tâm
    bbox_min: np.ndarray        # (3,) góc nhỏ nhất AABB
    bbox_max: np.ndarray        # (3,) góc lớn nhất AABB
    size: np.ndarray            # (3,) kích thước (dx, dy, dz)
    distance: float             # khoảng cách tới gốc toạ độ (ego)
    n_points: int               # số điểm trong cluster
    height: float               # chiều cao vật cản (z_max - z_min)


@dataclass
class DetectionResult:
    """Kết quả pipeline phát hiện vật cản."""
    # Point clouds qua từng bước
    raw_points: np.ndarray
    downsampled_points: np.ndarray
    ground_points: np.ndarray
    non_ground_points: np.ndarray
    # Ground plane
    ground_plane: np.ndarray    # [a, b, c, d]
    # Clusters
    clusters: list[ObstacleCluster] = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    # Timing
    t_downsample_ms: float = 0.0
    t_ground_ms: float = 0.0
    t_cluster_ms: float = 0.0
    t_bbox_ms: float = 0.0
    t_total_ms: float = 0.0
    # Config
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------
def voxel_downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, float]:
    """Voxel downsample, trả về (points_down, time_ms)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    t0 = time.perf_counter()
    pcd_down = pcd.voxel_down_sample(voxel_size)
    dt = (time.perf_counter() - t0) * 1000
    return np.asarray(pcd_down.points), dt


def remove_ground(points: np.ndarray, distance_threshold: float = 0.2,
                  ransac_n: int = 3, num_iterations: int = 1000
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Tách mặt đất bằng RANSAC plane fitting.

    Returns:
        non_ground (N1, 3), ground (N2, 3), plane_model [a,b,c,d], time_ms
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    t0 = time.perf_counter()
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    dt = (time.perf_counter() - t0) * 1000
    ground = points[inliers]
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    non_ground = points[mask]
    return non_ground, ground, np.array(plane_model), dt


def cluster_obstacles(points: np.ndarray, eps: float = 0.5,
                      min_points: int = 10
                      ) -> tuple[np.ndarray, float]:
    """DBSCAN clustering, trả về (labels, time_ms). Label -1 = noise."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    t0 = time.perf_counter()
    labels = np.array(pcd.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=False
    ))
    dt = (time.perf_counter() - t0) * 1000
    return labels, dt


def extract_clusters(points: np.ndarray, labels: np.ndarray) -> list[ObstacleCluster]:
    """Tạo ObstacleCluster cho mỗi cluster_id >= 0."""
    clusters = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # bỏ noise
    for cid in sorted(unique_labels):
        mask = labels == cid
        pts = points[mask]
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        center = pts.mean(axis=0)
        size = bbox_max - bbox_min
        distance = float(np.linalg.norm(center[:2]))  # khoảng cách 2D tới ego
        height = float(size[2])
        clusters.append(ObstacleCluster(
            cluster_id=int(cid),
            points=pts,
            center=center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            size=size,
            distance=distance,
            n_points=int(len(pts)),
            height=height,
        ))
    return clusters


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def detect_obstacles(
    points: np.ndarray,
    voxel_size: float = 0.1,
    distance_threshold: float = 0.2,
    eps: float = 0.5,
    min_points: int = 10,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_range: float = 80.0,
) -> DetectionResult:
    """Chạy toàn bộ pipeline phát hiện vật cản.

    Args:
        points: (N, >=3) point cloud thô
        voxel_size: kích thước voxel để downsample (m)
        distance_threshold: ngưỡng RANSAC cho ground removal (m)
        eps: epsilon cho DBSCAN (m)
        min_points: số điểm tối thiểu cho 1 cluster DBSCAN
        ransac_n: số điểm dùng trong mỗi iteration RANSAC
        num_iterations: số iteration RANSAC
        max_range: bỏ điểm xa hơn giá trị này (m)

    Returns:
        DetectionResult chứa toàn bộ kết quả intermediate và cuối cùng
    """
    raw = points[:, :3].copy()

    # Lọc NaN/Inf và range
    valid = np.isfinite(raw).all(axis=1)
    raw = raw[valid]
    ranges = np.linalg.norm(raw[:, :2], axis=1)
    raw = raw[ranges <= max_range]

    t_total_start = time.perf_counter()

    # Bước 1: Voxel downsample
    down, t_down = voxel_downsample(raw, voxel_size)

    # Bước 2: Ground removal
    non_ground, ground, plane, t_ground = remove_ground(
        down, distance_threshold, ransac_n, num_iterations
    )

    # Bước 3: Clustering
    if len(non_ground) > 0:
        labels, t_cluster = cluster_obstacles(non_ground, eps, min_points)
    else:
        labels = np.array([], dtype=int)
        t_cluster = 0.0

    # Bước 4: Extract bounding boxes
    t_bbox_start = time.perf_counter()
    clusters = extract_clusters(non_ground, labels) if len(non_ground) > 0 else []
    t_bbox = (time.perf_counter() - t_bbox_start) * 1000

    t_total = (time.perf_counter() - t_total_start) * 1000

    return DetectionResult(
        raw_points=raw,
        downsampled_points=down,
        ground_points=ground,
        non_ground_points=non_ground,
        ground_plane=plane,
        clusters=clusters,
        labels=labels,
        t_downsample_ms=t_down,
        t_ground_ms=t_ground,
        t_cluster_ms=t_cluster,
        t_bbox_ms=t_bbox,
        t_total_ms=t_total,
        config={
            "voxel_size": voxel_size,
            "distance_threshold": distance_threshold,
            "eps": eps,
            "min_points": min_points,
            "max_range": max_range,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pipeline phát hiện vật cản: voxel downsample → ground removal → DBSCAN → bbox"
    )
    ap.add_argument("--data-root", default="data/kitti_mini",
                    help="thư mục KITTI hoặc nuScenes")
    ap.add_argument("--frame", default="000011", help="frame id")
    ap.add_argument("--voxel-size", type=float, default=0.1, help="kích thước voxel (m)")
    ap.add_argument("--distance-threshold", type=float, default=0.2,
                    help="RANSAC distance threshold cho ground (m)")
    ap.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon (m)")
    ap.add_argument("--min-points", type=int, default=10, help="DBSCAN min points")
    ap.add_argument("--max-range", type=float, default=80.0, help="lọc điểm xa hơn (m)")
    args = ap.parse_args()

    points = load_points(args.data_root, args.frame)
    result = detect_obstacles(
        points,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        eps=args.eps,
        min_points=args.min_points,
        max_range=args.max_range,
    )

    print(f"Frame: {args.frame}")
    print(f"  Raw points:        {len(result.raw_points)}")
    print(f"  After downsample:  {len(result.downsampled_points)}")
    print(f"  Ground points:     {len(result.ground_points)}")
    print(f"  Non-ground points: {len(result.non_ground_points)}")
    print(f"  Clusters:          {len(result.clusters)}")
    print(f"  Ground plane:      [{', '.join(f'{v:.4f}' for v in result.ground_plane)}]")
    print(f"  Timing: downsample={result.t_downsample_ms:.1f}ms "
          f"ground={result.t_ground_ms:.1f}ms "
          f"cluster={result.t_cluster_ms:.1f}ms "
          f"bbox={result.t_bbox_ms:.1f}ms "
          f"total={result.t_total_ms:.1f}ms")
    if result.clusters:
        nearest = min(result.clusters, key=lambda c: c.distance)
        largest = max(result.clusters, key=lambda c: c.n_points)
        print(f"  Nearest obstacle:  {nearest.distance:.1f}m ({nearest.n_points} pts, "
              f"height={nearest.height:.2f}m)")
        print(f"  Largest cluster:   {largest.n_points} pts, {largest.distance:.1f}m away, "
              f"height={largest.height:.2f}m")


if __name__ == "__main__":
    main()
