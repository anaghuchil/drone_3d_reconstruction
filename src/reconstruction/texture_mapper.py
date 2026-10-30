import os
import time
import cv2
import xatlas
import trimesh
import pycolmap
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import distance_transform_edt, minimum_filter

try:
    import torch
except ImportError:
    torch = None


class MultiViewTextureMapper:
    @staticmethod
    def _load_colmap_cameras(dense_dir: str, max_views: int = 35):
        sparse_dir = os.path.join(dense_dir, "sparse")
        images_dir = os.path.join(dense_dir, "images")

        if not os.path.exists(sparse_dir) or not os.path.exists(images_dir):
            raise FileNotFoundError(
                f"[-] COLMAP dense workspace missing at {dense_dir}. Run dense reconstruction first."
            )

        rec = pycolmap.Reconstruction(sparse_dir)
        all_images = sorted(list(rec.images.values()), key=lambda im: im.name)

        if len(all_images) > max_views:
            step = len(all_images) / float(max_views)
            selected_images = [all_images[int(i * step)] for i in range(max_views)]
        else:
            selected_images = all_images

        views = []
        for img_meta in selected_images:
            img_path = os.path.join(images_dir, img_meta.name)
            if not os.path.exists(img_path):
                continue

            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h, w = rgb.shape[:2]

            cam = rec.cameras[img_meta.camera_id]
            model_name = str(cam.model).upper()
            params = cam.params

            if "PINHOLE" in model_name and "SIMPLE" not in model_name and len(params) >= 4:
                fx, fy, cx, cy = float(params[0]), float(params[1]), float(params[2]), float(params[3])
            else:
                fx = fy = float(params[0])
                cx = float(params[1]) if len(params) > 1 else w / 2.0
                cy = float(params[2]) if len(params) > 2 else h / 2.0

            cam_from_world = img_meta.cam_from_world
            if callable(cam_from_world):
                cam_from_world = cam_from_world()
            R = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float32)
            t = np.asarray(cam_from_world.translation, dtype=np.float32)
            center = -R.T @ t

            views.append({
                "name": img_meta.name,
                "rgb": rgb,
                "w": w,
                "h": h,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "R": R,
                "t": t,
                "center": center,
            })

        print(f"[+] Loaded {len(views)} calibrated COLMAP camera views for 4K texture projection.")
        return views

    @staticmethod
    def _subdivide_triangles_to_texels(
        v3d: np.ndarray,
        n3d: np.ndarray,
        uv: np.ndarray,
        faces: np.ndarray,
        atlas_size: int = 4096,
    ):
        tri_v = v3d[faces].astype(np.float32)
        tri_n = n3d[faces].astype(np.float32)
        tri_uv = uv[faces].astype(np.float32)

        completed_v, completed_n, completed_uv = [], [], []

        for _ in range(7):
            if len(tri_uv) == 0:
                break

            e0 = np.linalg.norm(tri_uv[:, 0] - tri_uv[:, 1], axis=1) * atlas_size
            e1 = np.linalg.norm(tri_uv[:, 1] - tri_uv[:, 2], axis=1) * atlas_size
            e2 = np.linalg.norm(tri_uv[:, 2] - tri_uv[:, 0], axis=1) * atlas_size
            max_edge = np.maximum(np.maximum(e0, e1), e2)

            small_mask = max_edge <= 1.15
            if np.any(small_mask):
                completed_v.append(tri_v[small_mask].reshape(-1, 3))
                completed_n.append(tri_n[small_mask].reshape(-1, 3))
                completed_uv.append(tri_uv[small_mask].reshape(-1, 2))

            large_mask = ~small_mask
            if not np.any(large_mask):
                tri_v = np.empty((0, 3, 3), dtype=np.float32)
                break

            tv = tri_v[large_mask]
            tn = tri_n[large_mask]
            tuv = tri_uv[large_mask]

            mv01 = 0.5 * (tv[:, 0] + tv[:, 1])
            mv12 = 0.5 * (tv[:, 1] + tv[:, 2])
            mv20 = 0.5 * (tv[:, 2] + tv[:, 0])

            mn01 = 0.5 * (tn[:, 0] + tn[:, 1])
            mn12 = 0.5 * (tn[:, 1] + tn[:, 2])
            mn20 = 0.5 * (tn[:, 2] + tn[:, 0])

            muv01 = 0.5 * (tuv[:, 0] + tuv[:, 1])
            muv12 = 0.5 * (tuv[:, 1] + tuv[:, 2])
            muv20 = 0.5 * (tuv[:, 2] + tuv[:, 0])

            tri_v = np.concatenate([
                np.stack([tv[:, 0], mv01, mv20], axis=1),
                np.stack([tv[:, 1], mv12, mv01], axis=1),
                np.stack([tv[:, 2], mv20, mv12], axis=1),
                np.stack([mv01, mv12, mv20], axis=1),
            ], axis=0)

            tri_n = np.concatenate([
                np.stack([tn[:, 0], mn01, mn20], axis=1),
                np.stack([tn[:, 1], mn12, mn01], axis=1),
                np.stack([tn[:, 2], mn20, mn12], axis=1),
                np.stack([mn01, mn12, mn20], axis=1),
            ], axis=0)

            tri_uv = np.concatenate([
                np.stack([tuv[:, 0], muv01, muv20], axis=1),
                np.stack([tuv[:, 1], muv12, muv01], axis=1),
                np.stack([tuv[:, 2], muv20, muv12], axis=1),
                np.stack([muv01, muv12, muv20], axis=1),
            ], axis=0)

            if len(tri_v) > 3500000:
                break

        if len(tri_v) > 0:
            completed_v.append(tri_v.reshape(-1, 3))
            completed_n.append(tri_n.reshape(-1, 3))
            completed_uv.append(tri_uv.reshape(-1, 2))

            completed_v.append(tri_v.mean(axis=1))
            completed_n.append(tri_n.mean(axis=1))
            completed_uv.append(tri_uv.mean(axis=1))

        pts_3d = np.concatenate(completed_v, axis=0)
        norms_3d = np.concatenate(completed_n, axis=0)
        uv_coords = np.concatenate(completed_uv, axis=0)

        cols = np.clip((uv_coords[:, 0] * (atlas_size - 1)).round().astype(np.int32), 0, atlas_size - 1)
        rows = np.clip(((1.0 - uv_coords[:, 1]) * (atlas_size - 1)).round().astype(np.int32), 0, atlas_size - 1)
        flat_idx = rows * atlas_size + cols

        _, unique_indices = np.unique(flat_idx, return_index=True)
        return (
            pts_3d[unique_indices],
            norms_3d[unique_indices],
            rows[unique_indices],
            cols[unique_indices],
        )

    @staticmethod
    def _delight_3d_texels(
        texel_rgb: np.ndarray,
        tex_norms: np.ndarray,
        observed_mask: np.ndarray,
        shadow_lift_strength: float = 0.65,
        sh_flatten_strength: float = 0.70,
    ) -> np.ndarray:
        if not np.any(observed_mask):
            return texel_rgb

        print("[+] Running 3D Spherical Harmonics & LAB Shadow-Removal Delighting Pass...")
        valid_rgb = np.clip(texel_rgb[observed_mask], 0.0, 1.0).astype(np.float32)
        valid_norms = tex_norms[observed_mask]

        lab = cv2.cvtColor(valid_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]

        mid_mask = (L > np.percentile(L, 15)) & (L < np.percentile(L, 90))
        if np.sum(mid_mask) > 500:
            H = np.column_stack([np.ones(np.sum(mid_mask), dtype=np.float32), valid_norms[mid_mask]])
            coeffs, _, _, _ = np.linalg.lstsq(H, L[mid_mask], rcond=None)
            H_all = np.column_stack([np.ones(len(L), dtype=np.float32), valid_norms])
            predicted_shading = H_all @ coeffs
            ambient_base = float(coeffs[0])
            shading_ratio = np.clip(predicted_shading / max(ambient_base, 10.0), 0.55, 1.45)
            L_sh_corrected = L / (1.0 + sh_flatten_strength * (shading_ratio - 1.0))
        else:
            L_sh_corrected = L.copy()

        med_L = float(np.percentile(L_sh_corrected, 60))
        std_L = float(np.std(L_sh_corrected)) + 1e-5
        med_B = float(np.median(B))

        shadow_luma_score = np.clip((med_L - L_sh_corrected) / (1.8 * std_L), 0.0, 1.0)
        cooler_tint_score = np.clip((med_B - B) / 15.0 + 0.5, 0.2, 1.0)
        shadow_prob = (shadow_luma_score ** 1.35) * cooler_tint_score

        target_lift = (med_L - L_sh_corrected) * shadow_lift_strength * shadow_prob
        L_delighted = np.clip(L_sh_corrected + target_lift, 0.0, 98.0)

        highlight_mask = L_delighted > 88.0
        if np.any(highlight_mask):
            L_delighted[highlight_mask] = 88.0 + (L_delighted[highlight_mask] - 88.0) * 0.45

        B_restored = B + (med_B - B) * (shadow_prob * 0.45 * shadow_lift_strength)
        chroma_boost = 1.0 + 0.20 * shadow_prob * shadow_lift_strength
        A_restored = np.clip(A * chroma_boost, -127.0, 127.0)
        B_restored = np.clip(B_restored * chroma_boost, -127.0, 127.0)

        delighted_lab = np.stack([L_delighted, A_restored, B_restored], axis=1).astype(np.float32)
        delighted_rgb = cv2.cvtColor(delighted_lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)

        out_rgb = texel_rgb.copy()
        out_rgb[observed_mask] = np.clip(delighted_rgb, 0.0, 1.0)
        return out_rgb

    @staticmethod
    def _generate_pbr_maps(
        atlas_rgb: np.ndarray,
        normal_strength: float = 2.6,
        base_roughness: float = 0.78,
    ):
        print("[+] Synthesizing 4K Tangent-Space Normal, Roughness, and Ambient Occlusion PBR maps...")
        rgb_float = atlas_rgb.astype(np.float32) / 255.0
        luma = (
            0.299 * rgb_float[..., 0]
            + 0.587 * rgb_float[..., 1]
            + 0.114 * rgb_float[..., 2]
        ).astype(np.float32)

        luma_fine = cv2.GaussianBlur(luma, (3, 3), 0.6)
        dx_fine = cv2.Sobel(luma_fine, cv2.CV_32F, 1, 0, ksize=3)
        dy_fine = cv2.Sobel(luma_fine, cv2.CV_32F, 0, 1, ksize=3)

        luma_meso = cv2.GaussianBlur(luma, (9, 9), 2.0)
        dx_meso = cv2.Sobel(luma_meso, cv2.CV_32F, 1, 0, ksize=3)
        dy_meso = cv2.Sobel(luma_meso, cv2.CV_32F, 0, 1, ksize=3)

        nx = -(0.65 * dx_fine + 0.35 * dx_meso) * normal_strength
        ny = (0.65 * dy_fine + 0.35 * dy_meso) * normal_strength
        nz = np.ones_like(luma, dtype=np.float32)

        norm_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

        normal_map_uint8 = np.zeros_like(atlas_rgb, dtype=np.uint8)
        normal_map_uint8[..., 0] = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        normal_map_uint8[..., 1] = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        normal_map_uint8[..., 2] = np.clip((nz * 0.5 + 0.5) * 255.0, 128, 255).astype(np.uint8)

        local_mean_luma = cv2.GaussianBlur(luma, (15, 15), 4.0)
        cavity_diff = np.clip(local_mean_luma - luma, 0.0, 0.35) / 0.35
        ao_float = np.clip(1.0 - 0.55 * cavity_diff, 0.35, 1.0)
        ao_map_uint8 = (ao_float * 255.0).astype(np.uint8)

        luma_sq_mean = cv2.GaussianBlur(luma * luma, (7, 7), 1.5)
        local_var = np.maximum(luma_sq_mean - (cv2.GaussianBlur(luma, (7, 7), 1.5) ** 2), 0.0)
        local_std = np.sqrt(local_var)
        norm_std = np.clip(local_std / (np.percentile(local_std, 95) + 1e-5), 0.0, 1.0)

        exg = np.clip(2.0 * rgb_float[..., 1] - rgb_float[..., 0] - rgb_float[..., 2], 0.0, 0.5) * 2.0
        roughness = (
            base_roughness
            + 0.16 * norm_std
            + 0.10 * exg
            + 0.08 * cavity_diff
            - 0.12 * np.clip(luma - 0.65, 0.0, 0.35)
        )
        roughness_uint8 = (np.clip(roughness, 0.40, 0.98) * 255.0).astype(np.uint8)

        orm_map_uint8 = np.zeros_like(atlas_rgb, dtype=np.uint8)
        orm_map_uint8[..., 0] = ao_map_uint8
        orm_map_uint8[..., 1] = roughness_uint8
        orm_map_uint8[..., 2] = 0

        return normal_map_uint8, orm_map_uint8, ao_map_uint8

    @classmethod
    def bake_4k_texture_atlas(
        cls,
        mesh_path: str,
        dense_dir: str = "data/outputs/sfm/dense",
        output_obj_path: str = "data/outputs/mesh/final_textured_model.obj",
        output_glb_path: str = "data/outputs/mesh/final_model.glb",
        atlas_size: int = 4096,
        target_faces: int = 85000,
        enable_delighting: bool = True,
    ) -> str:
        t0 = time.time()
        print(f"[+] Loading mesh for 4K xatlas UV unwrapping: {mesh_path}")

        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"[-] Mesh file not found: {mesh_path}")

        o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not o3d_mesh.has_triangles():
            raise RuntimeError("[-] Mesh contains no triangles.")

        has_vertex_colors = o3d_mesh.has_vertex_colors()

        if len(o3d_mesh.triangles) > target_faces:
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

        o3d_mesh.compute_vertex_normals()
        vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        normals = np.asarray(o3d_mesh.vertex_normals, dtype=np.float32)
        faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
        v_colors = (
            np.asarray(o3d_mesh.vertex_colors, dtype=np.float32)
            if has_vertex_colors
            else np.full_like(vertices, 0.65)
        )

        print("[+] Running xatlas UV parameterization...")
        vmapping, uv_faces, uvs = xatlas.parametrize(vertices, faces)
        uv_vertices = vertices[vmapping]
        uv_normals = normals[vmapping]
        uv_fallback_colors = v_colors[vmapping]

        print(f"[+] Rasterizing 3D world positions onto {atlas_size}x{atlas_size} UV grid...")
        tex_pts, tex_norms, tex_rows, tex_cols = cls._subdivide_triangles_to_texels(
            uv_vertices, uv_normals, uvs, uv_faces, atlas_size=atlas_size
        )
        num_texels = len(tex_pts)

        norm_len = np.linalg.norm(tex_norms, axis=1, keepdims=True)
        tex_norms = tex_norms / np.maximum(norm_len, 1e-8)

        views = cls._load_colmap_cameras(dense_dir=dense_dir, max_views=35)

        accum_color = np.zeros((num_texels, 3), dtype=np.float32)
        accum_weight = np.zeros(num_texels, dtype=np.float32)
        zbuf_scale = 4

        for view in views:
            R, t = view["R"], view["t"]
            fx, fy, cx, cy = view["fx"], view["fy"], view["cx"], view["cy"]
            w, h = view["w"], view["h"]
            cam_center = view["center"]

            pts_cam = tex_pts @ R.T + t
            z_cam = pts_cam[:, 2]

            view_dir = cam_center[None, :] - tex_pts
            dist = np.linalg.norm(view_dir, axis=1)
            cos_theta = np.sum(tex_norms * (view_dir / np.maximum(dist[:, None], 1e-6)), axis=1)

            in_front = (z_cam > 0.1) & (cos_theta > 0.15)
            if not np.any(in_front):
                continue

            u_proj = np.zeros(num_texels, dtype=np.float32)
            v_proj = np.zeros(num_texels, dtype=np.float32)
            u_proj[in_front] = (pts_cam[in_front, 0] / z_cam[in_front]) * fx + cx
            v_proj[in_front] = (pts_cam[in_front, 1] / z_cam[in_front]) * fy + cy

            in_bounds = (
                in_front
                & (u_proj >= 2.0)
                & (u_proj < (w - 3.0))
                & (v_proj >= 2.0)
                & (v_proj < (h - 3.0))
            )
            valid_idx = np.where(in_bounds)[0]
            if len(valid_idx) == 0:
                continue

            gw, gh = max(1, w // zbuf_scale), max(1, h // zbuf_scale)
            gx = np.clip((u_proj[valid_idx] / zbuf_scale).astype(np.int32), 0, gw - 1)
            gy = np.clip((v_proj[valid_idx] / zbuf_scale).astype(np.int32), 0, gh - 1)

            zbuf = np.full((gh, gw), np.inf, dtype=np.float32)
            np.minimum.at(zbuf, (gy, gx), z_cam[valid_idx])
            zbuf = minimum_filter(zbuf, size=3)

            vis_mask = z_cam[valid_idx] <= (zbuf[gy, gx] * 1.04)
            vis_idx = valid_idx[vis_mask]
            if len(vis_idx) == 0:
                continue

            # OpenCV cv2.remap has a hard limit of 32767 pixels per dimension (SHRT_MAX).
            # We process the texels in safe chunks to prevent assertion crashes.
            map_x = u_proj[vis_idx].astype(np.float32)
            map_y = v_proj[vis_idx].astype(np.float32)
            sampled_rgb = np.zeros((len(vis_idx), 3), dtype=np.float32)
            
            CHUNK_MAX = 30000
            for i in range(0, len(vis_idx), CHUNK_MAX):
                chunk_end = min(i + CHUNK_MAX, len(vis_idx))
                mx = map_x[i:chunk_end].reshape(1, -1)
                my = map_y[i:chunk_end].reshape(1, -1)
                remapped = cv2.remap(view["rgb"], mx, my, interpolation=cv2.INTER_LINEAR)
                sampled_rgb[i:chunk_end] = remapped.reshape(-1, 3)

            u_norm = (u_proj[vis_idx] - cx) / (w * 0.5)
            v_norm = (v_proj[vis_idx] - cy) / (h * 0.5)
            border_weight = np.clip(1.0 - 0.5 * (u_norm ** 2 + v_norm ** 2), 0.1, 1.0)
            weight = ((cos_theta[vis_idx] ** 6) / (z_cam[vis_idx] ** 2 + 1e-4)) * border_weight

            accum_color[vis_idx] += sampled_rgb * weight[:, None]
            accum_weight[vis_idx] += weight

        observed = accum_weight > 1e-9
        final_texel_colors = np.zeros((num_texels, 3), dtype=np.float32)
        final_texel_colors[observed] = accum_color[observed] / accum_weight[observed, None]

        if enable_delighting:
            final_texel_colors = cls._delight_3d_texels(
                texel_rgb=final_texel_colors,
                tex_norms=tex_norms,
                observed_mask=observed,
            )

        if np.any(~observed):
            mean_fallback = (
                final_texel_colors[observed].mean(axis=0)
                if np.any(observed)
                else uv_fallback_colors.mean(axis=0)
            )
            final_texel_colors[~observed] = mean_fallback

        atlas_rgb = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
        atlas_mask = np.zeros((atlas_size, atlas_size), dtype=bool)
        atlas_rgb[tex_rows, tex_cols] = (np.clip(final_texel_colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        atlas_mask[tex_rows, tex_cols] = True

        _, nearest_indices = distance_transform_edt(~atlas_mask, return_distances=True, return_indices=True)
        nr, nc = nearest_indices[0], nearest_indices[1]
        for c in range(3):
            atlas_rgb[..., c] = atlas_rgb[nr, nc, c]

        normal_map_uint8, orm_map_uint8, ao_map_uint8 = cls._generate_pbr_maps(atlas_rgb=atlas_rgb)

        base_prefix = os.path.splitext(output_obj_path)[0]
        albedo_png_path = f"{base_prefix}_4k_atlas.png"
        normal_png_path = f"{base_prefix}_4k_normal.png"
        roughness_png_path = f"{base_prefix}_4k_roughness.png"
        ao_png_path = f"{base_prefix}_4k_ao.png"

        pil_albedo = Image.fromarray(atlas_rgb, mode="RGB")
        pil_normal = Image.fromarray(normal_map_uint8, mode="RGB")
        pil_orm = Image.fromarray(orm_map_uint8, mode="RGB")
        pil_ao = Image.fromarray(ao_map_uint8, mode="L")

        pil_albedo.save(albedo_png_path, compress_level=3)
        pil_normal.save(normal_png_path, compress_level=3)
        pil_orm.save(roughness_png_path, compress_level=3)
        pil_ao.save(ao_png_path, compress_level=3)

        visuals = trimesh.visual.TextureVisuals(uv=uvs, image=pil_albedo)
        tm_mesh = trimesh.Trimesh(
            vertices=uv_vertices,
            faces=uv_faces,
            vertex_normals=uv_normals,
            visual=visuals,
            process=False,
        )
        tm_mesh.export(output_obj_path)

        centered_verts = uv_vertices.copy()
        centroid = centered_verts.mean(axis=0)
        centered_verts[:, 0] -= centroid[0]
        centered_verts[:, 2] -= centroid[2]
        centered_verts[:, 1] -= centered_verts[:, 1].min()

        pbr_material = trimesh.visual.material.PBRMaterial(
            name="AeroTerrain_4K_PBR",
            baseColorTexture=pil_albedo,
            normalTexture=pil_normal,
            metallicRoughnessTexture=pil_orm,
            occlusionTexture=pil_ao,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        )
        glb_visuals = trimesh.visual.TextureVisuals(uv=uvs, image=pil_albedo, material=pbr_material)
        glb_mesh = trimesh.Trimesh(
            vertices=centered_verts,
            faces=uv_faces,
            vertex_normals=uv_normals,
            visual=glb_visuals,
            process=False,
        )
        glb_mesh.export(output_glb_path, file_type="glb")

        elapsed = time.time() - t0
        print(f"[+] Full 4K PBR Baking complete in {elapsed:.1f}s -> {output_glb_path}")
        return output_glb_path


def apply_uv_texture_mapping(
    mesh_path: str,
    keyframes_dir: str = None,
    output_obj_path: str = "data/outputs/mesh/final_textured_model.obj",
    dense_dir: str = "data/outputs/sfm/dense",
    output_glb_path: str = "data/outputs/mesh/final_model.glb",
    enable_delighting: bool = True,
):
    return MultiViewTextureMapper.bake_4k_texture_atlas(
        mesh_path=mesh_path,
        dense_dir=dense_dir,
        output_obj_path=output_obj_path,
        output_glb_path=output_glb_path,
        atlas_size=4096,
        enable_delighting=enable_delighting,
    )
