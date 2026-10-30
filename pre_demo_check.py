import os
import sys
import importlib
import subprocess


def main():
    print("\n" + "=" * 72)
    print("  AETHER3D — PRE-DEMO SYSTEM & MODULE HEALTH CHECK")
    print("=" * 72)
    failures = 0

    try:
        import torch
        print(f"  [PASS] PyTorch CUDA   : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    except Exception as e:
        print(f"  [FAIL] PyTorch        : {e}")
        failures += 1

    for pkg in ("open3d", "pycolmap", "trimesh", "xatlas", "rasterio", "pyproj", "laspy", "scipy", "cv2", "ultralytics", "fastapi"):
        try:
            importlib.import_module(pkg)
            print(f"  [PASS] Library        : {pkg}")
        except Exception as e:
            print(f"  [FAIL] Library        : {pkg} ({e})")
            failures += 1

    for mod in (
        "src.config",
        "src.preprocessing.keyframe_extractor",
        "src.preprocessing.mask_generator",
        "src.preprocessing.metadata_parser",
        "src.preprocessing.georeferencing",
        "src.pose_estimation.sfm_engine",
        "src.reconstruction.mesh_generator",
        "src.reconstruction.texture_mapper",
        "src.reconstruction.pipeline_runner",
        "src.export.exporter",
        "src.export.export_las",
        "src.export.export_geotiff",
        "src.export.export_fbx",
        "src.export.terrain_analyzer",
    ):
        try:
            importlib.import_module(mod)
            print(f"  [PASS] Project Module : {mod}")
        except Exception as e:
            print(f"  [FAIL] Project Module : {mod} ({e})")
            failures += 1

    print("=" * 72)
    if failures == 0:
        print("  ✅ ALL MODULES & LIBRARIES VERIFIED READY!\n")
    else:
        print(f"  ❌ {failures} CHECK(S) FAILED — RUN pip install -r requirements.txt\n")


if __name__ == "__main__":
    main()
