import os
import laspy
import numpy as np
import open3d as o3d


def convert_ply_to_las(ply_path: str, las_path: str) -> str:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"[-] PLY file not found: {ply_path}")

    print(f"[+] Converting dense point cloud to LAS 1.2 format: {las_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float64)

    if len(points) == 0:
        print("[-] Warning: Point cloud is empty, skipping LAS export.")
        return None

    os.makedirs(os.path.dirname(las_path), exist_ok=True)

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    if pcd.has_colors():
        colors_u16 = (np.clip(np.asarray(pcd.colors), 0.0, 1.0) * 65535.0).astype(np.uint16)
        las.red = colors_u16[:, 0]
        las.green = colors_u16[:, 1]
        las.blue = colors_u16[:, 2]

    las.write(las_path)
    size_mb = os.path.getsize(las_path) / (1024 * 1024)
    print(f"[+] Exported LAS point cloud: {las_path} ({size_mb:.2f} MB)")
    return las_path
