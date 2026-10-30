import os
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


class ModelExporter:
    @staticmethod
    def convert_to_glb(obj_path: str) -> str:
        if trimesh is None:
            raise ImportError("[-] 'trimesh' package not found. Run: pip install trimesh")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"[-] Source mesh not found: {obj_path}")

        print(f"[+] Loading {obj_path} for grounding and double-sided GLB export...")
        mesh = trimesh.load(obj_path, process=False)

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if len(mesh.vertices) > 0:
            centroid = mesh.vertices.mean(axis=0)
            mesh.vertices[:, 0] -= centroid[0]
            mesh.vertices[:, 2] -= centroid[2]
            min_y = mesh.vertices[:, 1].min()
            mesh.vertices[:, 1] -= min_y

        if len(mesh.faces) > 0:
            flipped_faces = mesh.faces[:, ::-1]
            mesh.faces = np.vstack([mesh.faces, flipped_faces])

        glb_path = os.path.join(os.path.dirname(obj_path), "final_model.glb")
        mesh.export(glb_path, file_type="glb")

        if not os.path.exists(glb_path) or os.path.getsize(glb_path) == 0:
            raise RuntimeError(f"[-] GLB export failed: {glb_path}")

        file_size_mb = os.path.getsize(glb_path) / (1024 * 1024)
        print(f"[+] Exported grounded, double-sided model: {glb_path} ({file_size_mb:.2f} MB)")
        return glb_path
