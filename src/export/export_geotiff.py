import os
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
except ImportError:
    rasterio = None


class GeoTIFFExporter:
    @staticmethod
    def _align_to_nadir_plane(points: np.ndarray, normals: np.ndarray = None) -> np.ndarray:
        centered = points - np.mean(points, axis=0)
        if normals is not None and len(normals) == len(points):
            up_vec = np.mean(normals, axis=0)
            norm = np.linalg.norm(up_vec)
            up_vec = up_vec / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        else:
            cov = np.cov(centered, rowvar=False)
            _, eigvecs = np.linalg.eigh(cov)
            up_vec = eigvecs[:, 0]

        if up_vec[2] < 0:
            up_vec = -up_vec

        target_up = np.array([0.0, 0.0, 1.0])
        v = np.cross(up_vec, target_up)
        c = float(np.dot(up_vec, target_up))
        s = float(np.linalg.norm(v))

        if s > 1e-6:
            kmat = np.array([
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ])
            R = np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - c) / (s ** 2))
            aligned = centered @ R.T
        else:
            aligned = centered

        aligned[:, 2] -= np.percentile(aligned[:, 2], 1.0)
        return aligned

    @staticmethod
    def _generate_hillshade_rgb(dem: np.ndarray, valid_mask: np.ndarray, gsd: float) -> np.ndarray:
        smoothed_dem = gaussian_filter(dem, sigma=1.0)
        dy, dx = np.gradient(smoothed_dem, gsd)

        slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
        aspect = np.arctan2(-dx, dy)

        azimuth_rad = np.radians(315.0)
        altitude_rad = np.radians(45.0)

        shaded = (
            np.sin(altitude_rad) * np.sin(slope)
            + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
        )
        shaded = np.clip((shaded + 1.0) * 0.5, 0.0, 1.0)

        valid_z = dem[valid_mask]
        z_min = float(np.percentile(valid_z, 2)) if len(valid_z) > 0 else 0.0
        z_max = float(np.percentile(valid_z, 98)) if len(valid_z) > 0 else 1.0
        norm_z = np.clip((dem - z_min) / max(z_max - z_min, 1e-5), 0.0, 1.0)

        r = np.clip(norm_z * 1.3, 0.2, 1.0) * shaded
        g = np.clip(0.8 - np.abs(norm_z - 0.5) * 1.2, 0.2, 1.0) * shaded
        b = np.clip((1.0 - norm_z) * 0.9, 0.1, 1.0) * shaded

        rgba = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = (r * 255).astype(np.uint8)
        rgba[..., 1] = (g * 255).astype(np.uint8)
        rgba[..., 2] = (b * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)
        return rgba

    @classmethod
    def export_orthomosaic_and_dem(
        cls,
        ply_path: str,
        output_dir: str = "data/outputs/mesh",
        mesh_path: str = None,
        georef_json_path: str = None,
        grid_resolution: int = 2048,
        default_crs: str = "EPSG:32643",
        default_origin_utm: tuple = (275000.0, 2110000.0),
    ) -> dict:
        if rasterio is None:
            raise ImportError("[-] 'rasterio' is required. Install with: pip install rasterio pyproj")

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"[-] Dense point cloud not found: {ply_path}")

        os.makedirs(output_dir, exist_ok=True)
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_points():
            raise RuntimeError("[-] Cannot generate GeoTIFF from an empty point cloud.")

        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pts = np.asarray(pcd_clean.points)
        cols = (
            (np.asarray(pcd_clean.colors) * 255.0).clip(0, 255).astype(np.uint8)
            if pcd_clean.has_colors()
            else np.full((len(pts), 3), 180, dtype=np.uint8)
        )
        normals = np.asarray(pcd_clean.normals) if pcd_clean.has_normals() else None

        if mesh_path and os.path.exists(mesh_path):
            try:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                if len(mesh.triangles) > 0:
                    sampled_pcd = mesh.sample_points_uniformly(number_of_points=min(len(pts), 500000))
                    s_pts = np.asarray(sampled_pcd.points)
                    s_cols = (
                        (np.asarray(sampled_pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
                        if sampled_pcd.has_colors()
                        else np.full((len(s_pts), 3), 180, dtype=np.uint8)
                    )
                    pts = np.vstack([pts, s_pts])
                    cols = np.vstack([cols, s_cols])
                    normals = None
            except Exception:
                pass

        is_georeferenced = georef_json_path and os.path.exists(georef_json_path)
        if not is_georeferenced:
            pts = cls._align_to_nadir_plane(pts, normals)

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_y, max_y = float(np.min(y)), float(np.max(y))

        span_x = max(max_x - min_x, 1e-3)
        span_y = max(max_y - min_y, 1e-3)

        gsd = max(span_x, span_y) / float(grid_resolution - 1)
        width = int(np.ceil(span_x / gsd)) + 1
        height = int(np.ceil(span_y / gsd)) + 1

        z_order = np.argsort(z)
        x_sorted = x[z_order]
        y_sorted = y[z_order]
        z_sorted = z[z_order].astype(np.float32)
        cols_sorted = cols[z_order]

        col_idx = np.clip(((x_sorted - min_x) / gsd).astype(np.int32), 0, width - 1)
        row_idx = np.clip(((max_y - y_sorted) / gsd).astype(np.int32), 0, height - 1)

        dem_grid = np.full((height, width), np.nan, dtype=np.float32)
        ortho_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        occupied = np.zeros((height, width), dtype=bool)

        dem_grid[row_idx, col_idx] = z_sorted
        ortho_rgb[row_idx, col_idx] = cols_sorted
        occupied[row_idx, col_idx] = True

        distances, nearest_indices = distance_transform_edt(
            ~occupied, return_distances=True, return_indices=True
        )
        valid_footprint = distances <= 6.0

        nearest_rows, nearest_cols = nearest_indices[0], nearest_indices[1]
        dem_filled = np.where(
            valid_footprint, dem_grid[nearest_rows, nearest_cols], -9999.0
        ).astype(np.float32)

        ortho_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for c in range(3):
            ortho_rgba[..., c] = np.where(
                valid_footprint, ortho_rgb[nearest_rows, nearest_cols, c], 0
            )
        ortho_rgba[..., 3] = np.where(valid_footprint, 255, 0).astype(np.uint8)

        if is_georeferenced:
            west, north = min_x, max_y
            pixel_size = gsd
        else:
            scale_to_meters = 50.0 / max(span_x, span_y)
            pixel_size = gsd * scale_to_meters
            west, north = default_origin_utm[0], default_origin_utm[1]

        transform = from_origin(west, north, pixel_size, pixel_size)
        crs = CRS.from_string(default_crs)

        ortho_tif_path = os.path.join(output_dir, "orthomosaic.tif")
        dem_tif_path = os.path.join(output_dir, "dem_elevation.tif")

        with rasterio.open(
            ortho_tif_path, "w", driver="GTiff", height=height, width=width,
            count=4, dtype="uint8", crs=crs, transform=transform, compress="deflate",
        ) as dst:
            for band_i in range(4):
                dst.write(ortho_rgba[..., band_i], band_i + 1)

        with rasterio.open(
            dem_tif_path, "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs=crs, transform=transform, nodata=-9999.0, compress="deflate",
        ) as dst:
            dst.write(dem_filled, 1)

        ortho_png_path = os.path.join(output_dir, "orthomosaic_preview.png")
        dem_png_path = os.path.join(output_dir, "dem_hillshade_preview.png")

        Image.fromarray(ortho_rgba, mode="RGBA").save(ortho_png_path)
        hillshade_rgba = cls._generate_hillshade_rgb(
            np.where(valid_footprint, dem_filled, 0.0), valid_footprint, pixel_size
        )
        Image.fromarray(hillshade_rgba, mode="RGBA").save(dem_png_path)

        print(f"[+] Exported GeoTIFF Orthomosaic & DEM: {ortho_tif_path}, {dem_tif_path}")
        return {
            "orthomosaic_tif": ortho_tif_path,
            "dem_tif": dem_tif_path,
            "orthomosaic_preview_png": ortho_png_path,
            "dem_preview_png": dem_png_path,
            "resolution_px": f"{width}x{height}",
            "gsd_meters_per_px": round(float(pixel_size), 4),
            "crs": default_crs,
        }
