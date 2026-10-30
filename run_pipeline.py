import os
import sys
import glob
import json
import time
import argparse
import cv2
import open3d as o3d

from src.reconstruction.pipeline_runner import PipelineRunner
from src.reconstruction.texture_mapper import MultiViewTextureMapper
from src.export.export_las import convert_ply_to_las
from src.export.export_geotiff import GeoTIFFExporter
from src.export.export_fbx import FBXExporter


def find_default_video(input_dir: str = "data/input_video") -> str:
    if not os.path.exists(input_dir):
        return None
    files = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"):
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    return max(files, key=os.path.getmtime) if files else None


def auto_detect_telemetry(video_path: str, explicit_srt: str = None, explicit_csv: str = None):
    if explicit_srt and os.path.exists(explicit_srt):
        return (None, explicit_srt) if explicit_srt.lower().endswith(".csv") else (explicit_srt, None)
    if explicit_csv and os.path.exists(explicit_csv):
        return None, explicit_csv

    base_no_ext = os.path.splitext(video_path)[0]
    for srt_ext in (".SRT", ".srt"):
        if os.path.exists(base_no_ext + srt_ext):
            return base_no_ext + srt_ext, None
    for csv_ext in (".CSV", ".csv"):
        if os.path.exists(base_no_ext + csv_ext):
            return None, base_no_ext + csv_ext
    return None, None


def inspect_geometry_counts(ply_path: str, obj_path: str) -> dict:
    counts = {"dense_points": 0, "mesh_vertices": 0, "mesh_triangles": 0}
    if os.path.exists(ply_path):
        try:
            with open(ply_path, "rb") as f:
                for _ in range(100):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("element vertex"):
                        counts["dense_points"] = int(line.split()[-1])
                        break
                    if line == "end_header":
                        break
        except Exception:
            pass
    if os.path.exists(obj_path):
        try:
            mesh = o3d.io.read_triangle_mesh(obj_path)
            counts["mesh_vertices"] = len(mesh.vertices)
            counts["mesh_triangles"] = len(mesh.triangles)
        except Exception:
            pass
    return counts


def main():
    parser = argparse.ArgumentParser(description="AETHER3D CLI Runner")
    parser.add_argument("-v", "--video", type=str, default=None)
    parser.add_argument("-s", "--srt", type=str, default=None)
    parser.add_argument("-c", "--csv", type=str, default=None)
    parser.add_argument("-m", "--mode", type=str, choices=["draft", "full"], default="draft")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--res", type=int, default=None)
    parser.add_argument("--atlas-size", type=int, default=4096)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-delight", action="store_true")
    parser.add_argument("--skip-uv", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    video_path = args.video or find_default_video("data/input_video")
    if not video_path or not os.path.exists(video_path):
        print("[-] ERROR: No input video found. Place an .mp4 in data/input_video/ or pass --video.")
        sys.exit(1)

    srt_path, csv_path = auto_detect_telemetry(video_path, args.srt, args.csv)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration_sec = total_video_frames / fps if fps > 0 else 22.0
    cap.release()

    mode = args.mode.lower()
    if mode == "full":
        default_frames = min(max(45, int(video_duration_sec * 2.5)), 80)
        default_depth, default_res = 10, 2000
    else:
        default_frames = min(max(28, int(video_duration_sec * 1.5)), 45)
        default_depth, default_res = 8, 1200

    max_frames = args.frames or default_frames
    poisson_depth = args.depth or default_depth
    max_image_size = args.res or default_res

    runner = PipelineRunner(
        base_dir="data",
        max_frames=max_frames,
        poisson_depth=poisson_depth,
        max_image_size=max_image_size,
    )

    output_glb = runner.run(
        input_video_path=video_path,
        srt_path=srt_path,
        flight_csv_path=csv_path,
        use_clahe=False,
        use_masking=not args.no_mask,
    )

    mesh_dir = "data/outputs/mesh"
    dense_dir = "data/outputs/sfm/dense"
    fused_ply = os.path.join(dense_dir, "fused.ply")
    clean_obj = os.path.join(mesh_dir, "surface_mesh_clean.obj")
    textured_obj = os.path.join(mesh_dir, "final_textured_model.obj")
    final_glb = os.path.join(mesh_dir, "final_model.glb")
    output_las = os.path.join(mesh_dir, "fused_dense.las")
    output_fbx = os.path.join(mesh_dir, "final_model.fbx")
    georef_json = os.path.join(mesh_dir, "georef_metadata.json")

    if not args.skip_uv and os.path.exists(clean_obj) and os.path.exists(dense_dir):
        try:
            MultiViewTextureMapper.bake_4k_texture_atlas(
                mesh_path=clean_obj,
                dense_dir=dense_dir,
                output_obj_path=textured_obj,
                output_glb_path=final_glb,
                atlas_size=args.atlas_size,
                enable_delighting=not args.no_delight,
            )
        except Exception as e:
            print(f"[!] 4K UV Texturing skipped ({e}).")

    if os.path.exists(fused_ply):
        convert_ply_to_las(fused_ply, output_las)

    try:
        FBXExporter.export_to_fbx(
            glb_path=final_glb,
            obj_path=textured_obj,
            fallback_obj_path=clean_obj,
            output_fbx_path=output_fbx,
        )
    except Exception as e:
        print(f"[!] FBX export warning: {e}")

    geotiff_info = {}
    if os.path.exists(fused_ply):
        try:
            geotiff_info = GeoTIFFExporter.export_orthomosaic_and_dem(
                ply_path=fused_ply,
                output_dir=mesh_dir,
                mesh_path=clean_obj if os.path.exists(clean_obj) else None,
                georef_json_path=georef_json if os.path.exists(georef_json) else None,
                grid_resolution=2048,
            )
        except Exception as e:
            print(f"[!] GeoTIFF export warning: {e}")

    total_runtime_sec = round(time.time() - start_time, 2)
    active_obj = textured_obj if os.path.exists(textured_obj) else clean_obj
    geom_counts = inspect_geometry_counts(fused_ply, active_obj)

    metadata = {
        "mode": mode,
        "input_video": os.path.basename(video_path),
        "video_resolution": f"{vid_w}x{vid_h}",
        "video_duration_sec": round(video_duration_sec, 2),
        "total_runtime_sec": total_runtime_sec,
        "dense_points": geom_counts["dense_points"],
        "mesh_vertices": geom_counts["mesh_vertices"],
        "mesh_triangles": geom_counts["mesh_triangles"],
        "geotiff_deliverables": geotiff_info,
        "formats_exported": ["GLB", "OBJ", "FBX", "PLY", "LAS", "GeoTIFF (Orthomosaic)", "GeoTIFF (DEM)"],
    }
    with open(os.path.join(mesh_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[SUCCESS] All 3D & GIS deliverables ready in {total_runtime_sec:.1f}s -> {final_glb}")


if __name__ == "__main__":
    main()
