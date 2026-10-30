import os
import json
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Optional


class TerrainAnalyzer:
    _cached_mesh_path: Optional[str] = None
    _cached_mtime: float = 0.0
    _cached_vertices: Optional[np.ndarray] = None
    _cached_faces: Optional[np.ndarray] = None
    _cached_kdtree: Optional[cKDTree] = None

    @classmethod
    def _load_active_mesh(cls, glb_path: str, obj_path: str):
        target_path = glb_path if os.path.exists(glb_path) else obj_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"[-] No 3D mesh found at {glb_path} or {obj_path}.")

        mtime = os.path.getmtime(target_path)
        if (
            cls._cached_mesh_path == target_path
            and cls._cached_mtime == mtime
            and cls._cached_vertices is not None
        ):
            return cls._cached_vertices, cls._cached_faces, cls._cached_kdtree

        mesh = trimesh.load(target_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        kdtree = cKDTree(vertices)

        cls._cached_mesh_path = target_path
        cls._cached_mtime = mtime
        cls._cached_vertices = vertices
        cls._cached_faces = faces
        cls._cached_kdtree = kdtree
        return vertices, faces, kdtree

    @staticmethod
    def _resolve_metric_scale(vertices: np.ndarray, georef_json_path: str) -> tuple:
        if georef_json_path and os.path.exists(georef_json_path):
            try:
                with open(georef_json_path, "r") as f:
                    data = json.load(f)
                scale = float(data.get("estimated_scale_factor", 1.0))
                if scale > 1e-6:
                    return scale, True
            except Exception:
                pass
        span = np.ptp(vertices, axis=0)
        max_span = float(np.max(span)) if len(span) > 0 else 1.0
        if max_span < 25.0:
            return 60.0 / max(max_span, 1e-3), False
        return 1.0, False

    @classmethod
    def measure_between_points(
        cls,
        point_a: List[float],
        point_b: List[float],
        glb_path: str = "data/outputs/mesh/final_model.glb",
        obj_path: str = "data/outputs/mesh/surface_mesh_clean.obj",
        georef_json_path: str = "data/outputs/mesh/georef_metadata.json",
        corridor_width_m: Optional[float] = None,
        up_axis: str = "y",
    ) -> dict:
        vertices, _, _ = cls._load_active_mesh(glb_path, obj_path)
        scale_to_m, is_georef = cls._resolve_metric_scale(vertices, georef_json_path)

        pA_m = np.array(point_a, dtype=np.float64) * scale_to_m
        pB_m = np.array(point_b, dtype=np.float64) * scale_to_m
        verts_m = vertices * scale_to_m

        v_idx = 2 if up_axis.lower() == "z" else 1
        h_idx = [0, 1] if up_axis.lower() == "z" else [0, 2]

        dist_3d_m = float(np.linalg.norm(pB_m - pA_m))
        pA_horiz, pB_horiz = pA_m[h_idx], pB_m[h_idx]
        delta_horiz = pB_horiz - pA_horiz
        horiz_dist_m = float(np.linalg.norm(delta_horiz))

        if horiz_dist_m < 1e-4:
            raise ValueError("[-] Clicked points are vertically coincident.")

        elev_a_m, elev_b_m = float(pA_m[v_idx]), float(pB_m[v_idx])
        point_to_point_relief_m = abs(elev_b_m - elev_a_m)
        slope_deg = float(np.degrees(np.arctan2(point_to_point_relief_m, horiz_dist_m)))
        slope_grade_pct = float((point_to_point_relief_m / horiz_dist_m) * 100.0)

        u_hat = delta_horiz / horiz_dist_m
        v_hat = np.array([-u_hat[1], u_hat[0]], dtype=np.float64)
        swath_width_m = (
            float(np.clip(horiz_dist_m * 0.25, 2.0, 25.0))
            if not corridor_width_m or corridor_width_m <= 0
            else float(corridor_width_m)
        )

        nu, nv = 60, 24
        u_vals = np.linspace(0.0, horiz_dist_m, nu)
        v_vals = np.linspace(-swath_width_m * 0.5, swath_width_m * 0.5, nv)
        du = horiz_dist_m / float(nu - 1)
        dv = swath_width_m / float(nv - 1)
        uu, vv = np.meshgrid(u_vals, v_vals)

        grid_horiz_m = (
            pA_horiz[None, None, :]
            + uu[:, :, None] * u_hat[None, None, :]
            + vv[:, :, None] * v_hat[None, None, :]
        )

        horiz_tree = cKDTree(verts_m[:, h_idx])
        dists, idxs = horiz_tree.query(grid_horiz_m.reshape(-1, 2), k=min(4, len(verts_m)))
        if dists.ndim == 1:
            dists, idxs = dists[:, None], idxs[:, None]

        weights = 1.0 / (dists ** 2 + 1e-6)
        weights /= np.sum(weights, axis=1, keepdims=True)
        grid_elev_m = np.sum(verts_m[:, v_idx][idxs] * weights, axis=1).reshape(nv, nu)

        planimetric_area_m2 = float(horiz_dist_m * swath_width_m)
        dz_dv, dz_du = np.gradient(grid_elev_m, dv, du)
        true_surface_area_m2 = float(np.mean(np.sqrt(1.0 + dz_du ** 2 + dz_dv ** 2)) * planimetric_area_m2)

        min_elev_m, max_elev_m = float(np.min(grid_elev_m)), float(np.max(grid_elev_m))
        baseline_elev_u = elev_a_m + (u_vals / horiz_dist_m) * (elev_b_m - elev_a_m)
        diff_from_baseline = grid_elev_m - baseline_elev_u[None, :]
        cell_area_m2 = (horiz_dist_m / float(nu)) * (swath_width_m / float(nv))

        cut_debris_volume_m3 = float(np.sum(np.maximum(diff_from_baseline, 0.0)) * cell_area_m2)
        fill_void_volume_m3 = float(np.sum(np.maximum(-diff_from_baseline, 0.0)) * cell_area_m2)
        heap_volume_above_floor_m3 = float(np.sum(grid_elev_m - min_elev_m) * cell_area_m2)
        active_debris_volume_m3 = (
            cut_debris_volume_m3
            if cut_debris_volume_m3 >= 0.15 * heap_volume_above_floor_m3
            else heap_volume_above_floor_m3
        )

        center_row = nv // 2
        profile_indices = np.linspace(0, nu - 1, 25).round().astype(int)
        elevation_profile = [
            {
                "distance_m": round(float(u_vals[idx]), 2),
                "terrain_elevation_m": round(float(grid_elev_m[center_row, idx]), 3),
                "baseline_elevation_m": round(float(baseline_elev_u[idx]), 3),
                "debris_thickness_m": round(float(grid_elev_m[center_row, idx] - baseline_elev_u[idx]), 3),
            }
            for idx in profile_indices
        ]

        return {
            "is_georeferenced": is_georef,
            "scale_factor_applied": round(scale_to_m, 4),
            "corridor_width_m": round(swath_width_m, 2),
            "distances": {
                "euclidean_3d_distance_m": round(dist_3d_m, 3),
                "horizontal_ground_distance_m": round(horiz_dist_m, 3),
            },
            "surface_area": {
                "true_3d_surface_area_m2": round(true_surface_area_m2, 2),
                "planimetric_2d_area_m2": round(planimetric_area_m2, 2),
                "surface_roughness_ratio": round(true_surface_area_m2 / max(planimetric_area_m2, 1e-4), 4),
            },
            "elevation_relief": {
                "point_a_elevation_m": round(elev_a_m, 3),
                "point_b_elevation_m": round(elev_b_m, 3),
                "point_to_point_relief_m": round(point_to_point_relief_m, 3),
                "max_corridor_relief_m": round(max_elev_m - min_elev_m, 3),
                "average_slope_degrees": round(slope_deg, 2),
                "average_slope_grade_pct": round(slope_grade_pct, 2),
            },
            "volumetrics": {
                "debris_volume_m3": round(active_debris_volume_m3, 2),
                "cut_volume_above_grade_m3": round(cut_debris_volume_m3, 2),
                "fill_void_below_grade_m3": round(fill_void_volume_m3, 2),
                "estimated_debris_mass_tons": round(active_debris_volume_m3 * 1.8, 2),
                "estimated_clearance_time_hours": round(active_debris_volume_m3 / 60.0, 2),
            },
            "elevation_profile": elevation_profile,
        }
