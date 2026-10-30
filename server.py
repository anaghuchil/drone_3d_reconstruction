import os
import json
import time
import shutil
import socket
import asyncio
import threading
import cv2
import open3d as o3d
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.reconstruction.pipeline_runner import PipelineRunner
from src.reconstruction.texture_mapper import MultiViewTextureMapper
from src.export.export_las import convert_ply_to_las
from src.export.export_geotiff import GeoTIFFExporter
from src.export.export_fbx import FBXExporter
from src.export.terrain_analyzer import TerrainAnalyzer

app = FastAPI(title="AETHER3D Spatial Reconstruction API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "data"
INPUT_VIDEO_DIR = os.path.join(BASE_DIR, "input_video")
MESH_DIR = os.path.join(BASE_DIR, "outputs", "mesh")
DENSE_DIR = os.path.join(BASE_DIR, "outputs", "sfm", "dense")

for d in (INPUT_VIDEO_DIR, MESH_DIR, DENSE_DIR):
    os.makedirs(d, exist_ok=True)

DELIVERABLE_PATHS = {
    "glb": os.path.join(MESH_DIR, "final_model.glb"),
    "obj": os.path.join(MESH_DIR, "surface_mesh_clean.obj"),
    "textured_obj": os.path.join(MESH_DIR, "final_textured_model.obj"),
    "fbx": os.path.join(MESH_DIR, "final_model.fbx"),
    "ply": os.path.join(DENSE_DIR, "fused.ply"),
    "las": os.path.join(MESH_DIR, "fused_dense.las"),
    "orthomosaic_tif": os.path.join(MESH_DIR, "orthomosaic.tif"),
    "dem_tif": os.path.join(MESH_DIR, "dem_elevation.tif"),
    "ortho_preview": os.path.join(MESH_DIR, "orthomosaic_preview.png"),
    "dem_preview": os.path.join(MESH_DIR, "dem_hillshade_preview.png"),
    "pbr_albedo": os.path.join(MESH_DIR, "final_textured_model_4k_atlas.png"),
    "pbr_normal": os.path.join(MESH_DIR, "final_textured_model_4k_normal.png"),
    "pbr_roughness": os.path.join(MESH_DIR, "final_textured_model_4k_roughness.png"),
    "pbr_ao": os.path.join(MESH_DIR, "final_textured_model_4k_ao.png"),
    "metadata": os.path.join(MESH_DIR, "metadata.json"),
    "georef": os.path.join(MESH_DIR, "georef_metadata.json"),
}

PIPELINE_STATE = {
    "status": "completed" if os.path.exists(DELIVERABLE_PATHS["glb"]) else "idle",
    "mode": "full",
    "progress": 100 if os.path.exists(DELIVERABLE_PATHS["glb"]) else 0,
    "current_stage": "Ready" if os.path.exists(DELIVERABLE_PATHS["glb"]) else "Idle",
    "error": None,
    "logs": ["[SYSTEM] AETHER3D Spatial Server Online."],
}
STATE_LOCK = threading.Lock()


class SpatialMeasureRequest(BaseModel):
    point_a: List[float] = Field(...)
    point_b: List[float] = Field(...)
    corridor_width_m: Optional[float] = Field(None)
    up_axis: str = Field("y")


def add_log(msg: str):
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(entry)
    with STATE_LOCK:
        PIPELINE_STATE["logs"].append(entry)


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def build_deliverables_manifest(base_url: str) -> dict:
    manifest = {}
    ts = int(time.time())
    for key, path in DELIVERABLE_PATHS.items():
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 2) if exists else 0.0
        manifest[key] = {
            "available": exists,
            "filename": os.path.basename(path),
            "size_mb": size_mb,
            "url": f"{base_url}/api/v1/download/{key}?t={ts}" if exists else None,
        }
    return manifest


@app.get("/api/v1/status")
def get_pipeline_status(request: Request):
    base_url = str(request.base_url).rstrip("/")
    with STATE_LOCK:
        state_copy = dict(PIPELINE_STATE)
    state_copy["deliverables"] = build_deliverables_manifest(base_url)
    return state_copy


@app.get("/api/v1/metadata")
def get_project_metadata(request: Request):
    base_url = str(request.base_url).rstrip("/")
    metadata = {}
    if os.path.exists(DELIVERABLE_PATHS["metadata"]):
        try:
            with open(DELIVERABLE_PATHS["metadata"], "r") as f:
                metadata = json.load(f)
        except Exception:
            pass
    metadata["deliverables"] = build_deliverables_manifest(base_url)
    return metadata


@app.get("/api/v1/download/{file_key}")
def download_deliverable(file_key: str):
    if file_key not in DELIVERABLE_PATHS:
        raise HTTPException(status_code=404, detail=f"Unknown key '{file_key}'.")
    file_path = DELIVERABLE_PATHS[file_key]
    if file_key == "fbx" and not os.path.exists(file_path):
        FBXExporter.export_to_fbx(
            glb_path=DELIVERABLE_PATHS["glb"],
            obj_path=DELIVERABLE_PATHS["textured_obj"],
            fallback_obj_path=DELIVERABLE_PATHS["obj"],
            output_fbx_path=file_path,
        )
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_key}' not generated yet.")
    return FileResponse(path=file_path, filename=os.path.basename(file_path))


@app.post("/api/v1/measure")
def calculate_spatial_measurement(req: SpatialMeasureRequest):
    try:
        return TerrainAnalyzer.measure_between_points(
            point_a=req.point_a,
            point_b=req.point_b,
            glb_path=DELIVERABLE_PATHS["glb"],
            obj_path=DELIVERABLE_PATHS["obj"],
            georef_json_path=DELIVERABLE_PATHS["georef"],
            corridor_width_m=req.corridor_width_m,
            up_axis=req.up_axis,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.websocket("/ws/pipeline")
async def pipeline_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with STATE_LOCK:
                payload = dict(PIPELINE_STATE)
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print(f"\n[+] AETHER3D Server Live -> Local: http://127.0.0.1:8000 | Network: http://{local_ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
