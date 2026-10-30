import os
import json
import numpy as np
import open3d as o3d
import pandas as pd
from typing import Dict, Tuple


class Georeferencer:
    @staticmethod
    def lla_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        a = 6378137.0
        e_sq = 6.69437999014e-3
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        N = a / np.sqrt(1.0 - e_sq * (np.sin(lat_rad) ** 2))
        X = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        Y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (N * (1.0 - e_sq) + alt) * np.sin(lat_rad)
        return np.vstack([X, Y, Z]).T

    @classmethod
    def lla_to_local_enu(
        cls, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        ecef = cls.lla_to_ecef(lat, lon, alt)
        lat0, lon0, alt0 = float(lat[0]), float(lon[0]), float(alt[0])
        origin_ecef = ecef[0]

        phi = np.radians(lat0)
        lam = np.radians(lon0)

        R_enu = np.array([
            [-np.sin(lam), np.cos(lam), 0.0],
            [-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)],
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        ], dtype=np.float64)

        delta_ecef = ecef - origin_ecef[None, :]
        enu_pts = delta_ecef @ R_enu.T

        origin_meta = {
            "origin_latitude": lat0,
            "origin_longitude": lon0,
            "origin_altitude_m": alt0,
            "origin_ecef_m": origin_ecef.tolist(),
        }
        return enu_pts, origin_meta

    @staticmethod
    def compute_umeyama_transform(
        src_pts: np.ndarray, dst_pts: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        n = src_pts.shape[0]
        src_mean = src_pts.mean(axis=0)
        dst_mean = dst_pts.mean(axis=0)

        src_demean = src_pts - src_mean
        dst_demean = dst_pts - dst_mean

        var_src = float(np.sum(src_demean ** 2) / n)
        if var_src < 1e-9:
            return 1.0, np.eye(3), np.zeros(3), 0.0

        cov = (dst_demean.T @ src_demean) / n
        U, S, Vt = np.linalg.svd(cov)

        d = np.ones(3, dtype=np.float64)
        if np.linalg.det(U @ Vt) < 0:
            d[-1] = -1.0

        R = U @ np.diag(d) @ Vt
        scale = float(np.sum(S * d) / var_src)
        translation = dst_mean - scale * (R @ src_mean)

        transformed = scale * (src_pts @ R.T) + translation
        rmse = float(np.sqrt(np.mean(np.sum((transformed - dst_pts) ** 2, axis=1))))
        return scale, R, translation, rmse

    @classmethod
    def georeference_mesh(
        cls,
        mesh_path: str,
        camera_positions: Dict[str, np.ndarray],
        keyframe_timestamps: Dict[str, float],
        gps_df: pd.DataFrame,
        output_meta_path: str,
        max_time_diff_sec: float = 1.5,
    ) -> str:
        if gps_df is None or gps_df.empty or not camera_positions:
            print("[!] Skipping georeferencing (no GPS telemetry or camera poses available).")
            return mesh_path

        if not os.path.exists(mesh_path):
            return mesh_path

        gps_times = gps_df["timestamp_sec"].to_numpy()
        gps_enu, origin_meta = cls.lla_to_local_enu(
            gps_df["latitude"].to_numpy(),
            gps_df["longitude"].to_numpy(),
            gps_df["altitude"].to_numpy(),
        )

        src_list, dst_list = [], []
        for filename, local_pos in camera_positions.items():
            ts = keyframe_timestamps.get(filename)
            if ts is None:
                continue
            diffs = np.abs(gps_times - ts)
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] <= max_time_diff_sec:
                src_list.append(local_pos)
                dst_list.append(gps_enu[best_idx])

        n_matched = len(src_list)
        if n_matched < 3:
            print(f"[!] Only {n_matched} GPS correspondences found (need >= 3). Skipping georef.")
            return mesh_path

        src_pts = np.array(src_list, dtype=np.float64)
        dst_pts = np.array(dst_list, dtype=np.float64)

        scale, R, translation, rmse = cls.compute_umeyama_transform(src_pts, dst_pts)

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) > 0:
            aligned_verts = scale * (vertices @ R.T) + translation
            mesh.vertices = o3d.utility.Vector3dVector(aligned_verts)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(
                mesh_path, mesh, write_vertex_normals=True, write_vertex_colors=True
            )

        metrics = {
            "num_correspondences": n_matched,
            "estimated_scale_factor": round(scale, 6),
            "fit_rmse_meters": round(rmse, 4),
            "coordinate_system": "Local_ENU_Meters (WGS84 Anchored)",
            "origin_wgs84": origin_meta,
            "rotation_matrix": R.tolist(),
            "translation_vector_m": translation.tolist(),
        }
        with open(output_meta_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"[+] 7-DOF Georeferencing complete ({n_matched} cameras, scale={scale:.4f}, RMSE={rmse:.3f}m).")
        return mesh_path
