import os
import time
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class SurfaceMeshGenerator:
    def __init__(self, output_dir="data/outputs/mesh"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _get_horizontal_axes(pts: np.ndarray):
        """Identifies the two dominant horizontal terrain axes via PCA covariance."""
        centered = pts - np.mean(pts, axis=0)
        cov = np.cov(centered, rowvar=False)
        _, eigvecs = np.linalg.eigh(cov)
        # eigvecs[:, 1] and eigvecs[:, 2] are the two largest-variance (horizontal) axes
        return eigvecs[:, 1:3], np.mean(pts, axis=0)

    def generate_poisson_mesh(self, ply_path: str, depth: int = 10) -> str:
        t_start = time.time()
        print(f"[+] Loading dense point cloud from {ply_path}...")
        pcd = o3d.io.read_point_cloud(ply_path)

        if not pcd.has_points():
            raise RuntimeError("[-] Point cloud is empty.")

        original_count = len(pcd.points)
        print(f"[+] Loaded {original_count:,} dense points. Running gentle Statistical Outlier Removal...")
        pcd_clean, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.4)
        removed = original_count - len(inlier_idx)
        print(f"[+] Removed {removed:,} isolated noise points. {len(pcd_clean.points):,} points remain.")

        distances = pcd_clean.compute_nearest_neighbor_distance()
        avg_dist = float(np.percentile(distances, 75)) if len(distances) > 0 else 0.05

        if pcd_clean.has_normals():
            print("[+] Preserving COLMAP's native multi-view stereo surface normals...")
            pcd_clean.normalize_normals()
        else:
            print("[+] Estimating surface normals...")
            pcd_clean.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 5.0, max_nn=35)
            )
            pcd_clean.orient_normals_towards_camera_location(
                pcd_clean.get_center() + np.array([0.0, 0.0, 50.0])
            )

        print(f"[+] Running Screened Poisson Surface Reconstruction (Depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=depth, scale=1.12, linear_fit=True
        )

        densities = np.asarray(densities)
        if len(densities) == len(mesh.vertices):
            density_threshold = np.quantile(densities, 0.005)
            mesh.remove_vertices_by_mask(densities < density_threshold)

        pcd_pts = np.asarray(pcd_clean.points)
        mesh_pts = np.asarray(mesh.vertices)

        if len(pcd_pts) > 0 and len(mesh_pts) > 0:
            print("[+] Computing Footprint-Aware KD-Tree mask & K=4 color projection...")
            tree_3d = cKDTree(pcd_pts)
            k = min(4, len(pcd_pts))
            dists_3d, idxs = tree_3d.query(mesh_pts, k=k)
            if k == 1:
                dists_3d = dists_3d[:, None]
                idxs = idxs[:, None]

            if pcd_clean.has_colors():
                pcd_cols = np.asarray(pcd_clean.colors)
                eps = 1e-8
                weights = 1.0 / (dists_3d ** 2 + eps)
                weights /= np.sum(weights, axis=1, keepdims=True)
                interpolated_colors = np.sum(pcd_cols[idxs] * weights[:, :, np.newaxis], axis=1)
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(interpolated_colors, 0.0, 1.0))

            horiz_basis, center_pt = self._get_horizontal_axes(pcd_pts)
            pcd_2d = (pcd_pts - center_pt) @ horiz_basis
            mesh_2d = (mesh_pts - center_pt) @ horiz_basis

            tree_2d = cKDTree(pcd_2d)
            dists_2d, _ = tree_2d.query(mesh_2d, k=1)

            outer_fringe_mask = dists_2d > (avg_dist * 12.0)
            extreme_dome_mask = dists_3d[:, 0] > (avg_dist * 35.0)
            bubble_mask = outer_fringe_mask | extreme_dome_mask

            print(f"[+] Trimming {int(np.sum(bubble_mask)):,} outer perimeter bubble vertices...")
            mesh.remove_vertices_by_mask(bubble_mask)

        if len(mesh.triangles) > 0:
            print("[+] Filtering out disconnected floating mesh fragments...")
            tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
            tri_clusters = np.asarray(tri_clusters)
            cluster_n_tris = np.asarray(cluster_n_tris)
            min_cluster_size = max(150, int(len(mesh.triangles) * 0.005))
            triangles_to_remove = cluster_n_tris[tri_clusters] < min_cluster_size
            mesh.remove_triangles_by_mask(triangles_to_remove)
            mesh.remove_unreferenced_vertices()

        # 4. Light vertex-only smoothing (1 iteration) to blend surfaces seamlessly
        mesh = mesh.filter_smooth_simple(
            number_of_iterations=1,
            filter_scope=o3d.geometry.FilterScope.Vertex,
        )

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        obj_path = os.path.join(self.output_dir, "surface_mesh_clean.obj")
        o3d.io.write_triangle_mesh(
            obj_path, mesh, write_vertex_normals=True, write_vertex_colors=True
        )

        if not os.path.exists(obj_path) or os.path.getsize(obj_path) == 0:
            raise RuntimeError(f"[-] Output mesh missing or empty: {obj_path}")

        elapsed = time.time() - t_start
        print(
            f"[+] Complete hole-free mesh generated in {elapsed:.1f}s "
            f"({len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles)."
        )
        return obj_path