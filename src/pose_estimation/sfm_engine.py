import os
import time
import threading
import subprocess
import logging
import numpy as np
import pycolmap
from typing import Optional, Dict, Tuple


class SfMEngine:
    def __init__(
        self,
        output_dir="data/outputs/sfm",
        colmap_exe=r"C:\COLMAP\COLMAP.bat",
        max_image_size=2000,
    ):
        self.output_dir = output_dir
        self.dense_dir = os.path.join(output_dir, "dense")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dense_dir, exist_ok=True)
        self.database_path = os.path.join(self.output_dir, "database.db")
        self.colmap_exe = colmap_exe
        self.requested_max_image_size = max_image_size
        self.stage_timings = {}

    @staticmethod
    def get_gpu_vram_gb() -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
            )
            vram_mb = float(out.strip().split("\n")[0])
            return vram_mb / 1024.0
        except Exception as e:
            logging.warning(f"[GPU CHECK] Could not query nvidia-smi ({e}). Defaulting to 8.0 GB.")
            return 8.0

    def get_safe_max_image_size(self, vram_gb: float) -> int:
        if vram_gb <= 6.0:
            safe_cap = 1600
        elif vram_gb <= 8.5:
            safe_cap = 2000
        elif vram_gb <= 12.0:
            safe_cap = 2400
        else:
            safe_cap = self.requested_max_image_size
        return min(self.requested_max_image_size, safe_cap)

    def _run_colmap_command(self, args, step_name, timeout_sec=3600):
        print(f"[+] Running COLMAP step: {step_name}")
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

        start = time.time()
        output_lines = []

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
            )
        except FileNotFoundError:
            raise RuntimeError(f"[-] Could not find COLMAP executable at '{self.colmap_exe}'.")

        def _reader():
            for line in process.stdout:
                print(line, end="")
                output_lines.append(line)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()
        reader_thread.join(timeout=timeout_sec)

        if reader_thread.is_alive():
            process.kill()
            process.wait()
            raise RuntimeError(f"[-] COLMAP step '{step_name}' timed out after {timeout_sec}s.")

        process.wait()
        elapsed = time.time() - start
        self.stage_timings[step_name] = elapsed

        if process.returncode != 0:
            tail = "".join(output_lines[-40:])
            raise RuntimeError(
                f"[-] COLMAP step '{step_name}' failed (code {process.returncode}).\n{tail}"
            )

        print(f"[+] '{step_name}' completed in {elapsed:.1f}s")
        return output_lines

    @staticmethod
    def _count_ply_vertices(ply_path):
        try:
            with open(ply_path, "rb") as f:
                for _ in range(200):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("element vertex"):
                        return int(line.split()[-1])
                    if line == "end_header":
                        break
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_camera_positions(reconstruction) -> Dict[str, np.ndarray]:
        positions = {}
        for image in reconstruction.images.values():
            try:
                center = np.array(image.projection_center())
            except AttributeError:
                cam_from_world = image.cam_from_world
                if callable(cam_from_world):
                    cam_from_world = cam_from_world()
                R_cw = np.array(cam_from_world.rotation.matrix())
                t_cw = np.array(cam_from_world.translation)
                center = -R_cw.T @ t_cw
            positions[image.name] = center
        return positions

    def run_reconstruction(
        self,
        image_dir: str,
        color_image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        focal_length_px: Optional[float] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        color_image_dir = color_image_dir or image_dir

        if os.path.exists(self.database_path):
            os.remove(self.database_path)

        t0 = time.time()
        print("[+] Step 1/4: Running Sparse Reconstruction (SfM)...")

        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "SIMPLE_RADIAL"
        
        # Backwards compatibility for older pycolmap versions
        if hasattr(reader_options, "single_camera"):
            reader_options.single_camera = True

        if mask_dir and os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0:
            reader_options.mask_path = mask_dir
            print(f"[+] Applying YOLO dynamic-object masks from: {mask_dir}")

        if focal_length_px and image_width and image_height:
            cx, cy = image_width / 2.0, image_height / 2.0
            reader_options.camera_params = f"{focal_length_px},{cx},{cy},0.0"

        # Prepare extraction arguments safely
        extract_kwargs = {
            "database_path": self.database_path,
            "image_path": image_dir,
            "reader_options": reader_options,
        }
        
        # New pycolmap versions use CameraMode enum
        if hasattr(pycolmap, "CameraMode") and hasattr(pycolmap.CameraMode, "SINGLE"):
            extract_kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE

        pycolmap.extract_features(**extract_kwargs)

        num_keyframes = len(
            [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

        # Use exhaustive matching for <= 85 frames to guarantee full loop & wide-baseline connections
        if num_keyframes <= 85:
            print(f"[+] Running exhaustive feature matching across {num_keyframes} keyframes...")
            pycolmap.match_exhaustive(database_path=self.database_path)
        else:
            print(f"[+] Running wide-window sequential matching (overlap=20)...")
            if hasattr(pycolmap, "SequentialPairingOptions"):
                pairing_opts = pycolmap.SequentialPairingOptions()
                pairing_opts.overlap = 20
                pycolmap.match_sequential(
                    database_path=self.database_path, pairing_options=pairing_opts
                )
            else:
                pycolmap.match_sequential(database_path=self.database_path)

        reconstructions = pycolmap.incremental_mapping(
            database_path=self.database_path,
            image_path=image_dir,
            output_path=self.output_dir,
        )

        if not reconstructions:
            raise RuntimeError("[-] Sparse SfM failed. Keyframe overlap insufficient.")

        best_rec_id = max(reconstructions, key=lambda k: len(reconstructions[k].images))
        best_rec = reconstructions[best_rec_id]
        print(
            f"[+] Selected primary reconstruction '{best_rec_id}' "
            f"({len(best_rec.images)}/{num_keyframes} registered images, "
            f"{len(best_rec.points3D)} sparse points)."
        )

        camera_positions = self._extract_camera_positions(best_rec)

        sparse_model_dir = os.path.join(self.output_dir, str(best_rec_id))
        if not os.path.exists(sparse_model_dir):
            sparse_model_dir = self.output_dir

        self.stage_timings["sparse_reconstruction"] = time.time() - t0

        self._run_colmap_command(
            [
                self.colmap_exe,
                "image_undistorter",
                "--image_path",
                color_image_dir,
                "--input_path",
                sparse_model_dir,
                "--output_path",
                self.dense_dir,
                "--output_type",
                "COLMAP",
            ],
            "image_undistorter",
            timeout_sec=900,
        )

        vram_gb = self.get_gpu_vram_gb()
        safe_max_size = self.get_safe_max_image_size(vram_gb)
        print(
            f"[+] RTX GPU Detected ({vram_gb:.1f} GB VRAM) -> Running PatchMatch Stereo at {safe_max_size}px"
        )

        self._run_colmap_command(
            [
                self.colmap_exe,
                "patch_match_stereo",
                "--workspace_path",
                self.dense_dir,
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.max_image_size",
                str(safe_max_size),
                "--PatchMatchStereo.window_radius",
                "5",
                "--PatchMatchStereo.window_step",
                "1",
                "--PatchMatchStereo.num_samples",
                "15",
                "--PatchMatchStereo.num_iterations",
                "5",
                "--PatchMatchStereo.geom_consistency",
                "true",
                "--PatchMatchStereo.filter",
                "true",
                "--PatchMatchStereo.filter_min_ncc",
                "0.1",
                "--PatchMatchStereo.cache_size",
                "6",
                "--PatchMatchStereo.gpu_index",
                "0",
            ],
            "patch_match_stereo",
            timeout_sec=3600,
        )

        fused_ply_path = os.path.join(self.dense_dir, "fused.ply")
        self._run_colmap_command(
            [
                self.colmap_exe,
                "stereo_fusion",
                "--workspace_path",
                self.dense_dir,
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--StereoFusion.min_num_pixels",
                "3",
                "--StereoFusion.max_reproj_error",
                "2.0",
                "--output_path",
                fused_ply_path,
            ],
            "stereo_fusion",
            timeout_sec=900,
        )

        point_count = self._count_ply_vertices(fused_ply_path)
        if point_count is not None:
            print(f"[+] Dense fusion produced {point_count:,} points in fused.ply")
            if point_count < 5000:
                print(
                    "[!] WARNING: Low point count detected. Check if input video has sufficient camera movement."
                )

        return fused_ply_path, camera_positions