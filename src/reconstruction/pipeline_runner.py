import os
import gc
import time
import shutil
import torch

from src.config import PipelineConfig
from src.preprocessing.keyframe_extractor import KeyframeExtractor
from src.preprocessing.mask_generator import DynamicObjectMasker
from src.preprocessing.illumination import IlluminationEnhancer
from src.preprocessing.metadata_parser import MetadataParser
from src.preprocessing.georeferencing import Georeferencer
from src.pose_estimation.sfm_engine import SfMEngine
from src.reconstruction.mesh_generator import SurfaceMeshGenerator
from src.export.exporter import ModelExporter


class PipelineRunner:
    def __init__(
        self,
        base_dir="data",
        video_path=None,
        max_frames=65,
        poisson_depth=10,
        max_image_size=2000,
        **kwargs,
    ):
        self.base_dir = base_dir
        self.video_path = video_path

        self.max_frames, self.poisson_depth = PipelineConfig.sanitize_parameters(
            max_frames=max_frames,
            poisson_depth=poisson_depth,
        )
        self.max_image_size = max_image_size

        self.keyframes_dir = os.path.join(base_dir, "keyframes")
        self.clahe_dir = os.path.join(base_dir, "keyframes_clahe")
        self.masks_dir = os.path.join(base_dir, "masks")
        self.sfm_dir = os.path.join(base_dir, "outputs", "sfm")
        self.mesh_dir = os.path.join(base_dir, "outputs", "mesh")

        self.extractor = KeyframeExtractor(max_frames=self.max_frames)
        self.sfm_engine = SfMEngine(
            output_dir=self.sfm_dir,
            max_image_size=self.max_image_size,
        )
        self.mesh_generator = SurfaceMeshGenerator(output_dir=self.mesh_dir)
        self.stage_timings = {}

    def _clear_directory(self, path: str):
        gc.collect()
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except PermissionError:
                time.sleep(0.5)
                shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _time_stage(self, name: str, fn, *args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        self.stage_timings[name] = time.time() - start
        return result

    def _print_timing_summary(self, total_time: float):
        print("\n" + "=" * 54)
        print("               STAGE TIMING SUMMARY               ")
        print("=" * 54)
        for stage, duration in self.stage_timings.items():
            mins = duration / 60.0
            print(f"  • {stage:<26}: {duration:>7.2f}s ({mins:.2f} min)")
        print("-" * 54)
        print(f"  • TOTAL CORE RUNTIME        : {total_time:>7.2f}s ({total_time / 60.0:.2f} min)")
        print("=" * 54 + "\n")

    def run(
        self,
        input_video_path=None,
        srt_path=None,
        flight_csv_path=None,
        focal_length_px=None,
        image_width=None,
        image_height=None,
        use_clahe=False,
        use_masking=True,
        progress_callback=None,
    ) -> str:
        start_time = time.time()
        active_video_path = input_video_path if input_video_path else self.video_path
        if not active_video_path or not os.path.exists(active_video_path):
            raise FileNotFoundError(f"[-] Input video not found: {active_video_path}")

        print("[+] Resetting working session directories...")
        self._clear_directory(self.keyframes_dir)
        self._clear_directory(self.clahe_dir)
        self._clear_directory(self.masks_dir)
        self._clear_directory(self.sfm_dir)
        self._clear_directory(self.mesh_dir)

        gps_df = (
            MetadataParser.parse_dji_srt(srt_path)
            if srt_path
            else MetadataParser.parse_flight_csv(flight_csv_path)
        )

        if progress_callback:
            progress_callback(0.05, "Extracting sharp parallax keyframes...")
        extracted_frames = self._time_stage(
            "keyframe_extraction",
            self.extractor.extract_keyframes,
            video_path=active_video_path,
            output_dir=self.keyframes_dir,
        )
        self._clear_memory()

        keyframe_timestamps = {f["filename"]: f["timestamp_sec"] for f in extracted_frames}

        active_image_dir = self.keyframes_dir
        if use_clahe:
            if progress_callback:
                progress_callback(0.12, "Applying CLAHE illumination normalization...")
            active_image_dir = self._time_stage(
                "clahe_enhancement",
                IlluminationEnhancer.apply_clahe_preprocessing,
                input_dir=self.keyframes_dir,
                clahe_output_dir=self.clahe_dir,
            )

        active_mask_dir = None
        if use_masking:
            if progress_callback:
                progress_callback(0.18, "Masking dynamic objects (YOLOv8-Seg)...")
            try:
                masker = DynamicObjectMasker()
                self._time_stage(
                    "yolo_masking",
                    masker.generate_masks,
                    images_dir=self.keyframes_dir,
                    masks_dir=self.masks_dir,
                    progress_callback=progress_callback,
                    progress_range=(0.18, 0.30),
                )
                active_mask_dir = self.masks_dir
                del masker
                self._clear_memory()
            except Exception as e:
                print(f"[!] Skipping masking stage: {e}")

        if progress_callback:
            progress_callback(0.35, "Running COLMAP Sparse SfM & GPU Dense Stereo...")
        fused_ply_path, camera_positions = self._time_stage(
            "sfm_and_dense_stereo",
            self.sfm_engine.run_reconstruction,
            image_dir=active_image_dir,
            color_image_dir=self.keyframes_dir,
            mask_dir=active_mask_dir,
            focal_length_px=focal_length_px,
            image_width=image_width,
            image_height=image_height,
        )
        self._clear_memory()

        if progress_callback:
            progress_callback(0.80, "Generating Screened Poisson 3D Surface Mesh...")
        mesh_obj_path = self._time_stage(
            "mesh_generation",
            self.mesh_generator.generate_poisson_mesh,
            ply_path=fused_ply_path,
            depth=self.poisson_depth,
        )
        self._clear_memory()

        if gps_df is not None and not gps_df.empty:
            if progress_callback:
                progress_callback(0.88, "Applying 7-DOF Umeyama Georeferencing...")
            georef_meta_json = os.path.join(self.mesh_dir, "georef_metadata.json")
            self._time_stage(
                "georeferencing",
                Georeferencer.georeference_mesh,
                mesh_path=mesh_obj_path,
                camera_positions=camera_positions,
                keyframe_timestamps=keyframe_timestamps,
                gps_df=gps_df,
                output_meta_path=georef_meta_json,
            )

        if progress_callback:
            progress_callback(0.94, "Exporting grounded double-sided GLB...")
        glb_path = self._time_stage(
            "glb_export",
            ModelExporter.convert_to_glb,
            mesh_obj_path,
        )

        total_elapsed = time.time() - start_time
        self._print_timing_summary(total_elapsed)

        if progress_callback:
            progress_callback(1.0, "Core 3D Reconstruction Complete!")

        return glb_path
