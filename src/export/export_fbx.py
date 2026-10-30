import os
import glob
import time
import shutil
import tempfile
import subprocess
import trimesh
import numpy as np


class FBXExporter:
    @staticmethod
    def _find_blender_executable() -> str:
        blender_in_path = shutil.which("blender")
        if blender_in_path:
            return blender_in_path
        search_patterns = [
            r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
            r"C:\Program Files (x86)\Blender Foundation\Blender*\blender.exe",
        ]
        for pattern in search_patterns:
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                return matches[0]
        return None

    @classmethod
    def _export_via_headless_blender(cls, blender_exe: str, src_path: str, fbx_path: str) -> bool:
        abs_src = os.path.abspath(src_path).replace("\\", "/")
        abs_fbx = os.path.abspath(fbx_path).replace("\\", "/")
        ext = os.path.splitext(abs_src)[1].lower()

        script_lines = [
            "import bpy",
            "bpy.ops.wm.read_factory_settings(use_empty=True)",
            f"if '{ext}' in ['.glb', '.gltf']:",
            f"    bpy.ops.import_scene.gltf(filepath='{abs_src}')",
            "else:",
            "    try:",
            f"        bpy.ops.wm.obj_import(filepath='{abs_src}')",
            "    except AttributeError:",
            f"        bpy.ops.import_scene.obj(filepath='{abs_src}')",
            f"bpy.ops.export_scene.fbx(filepath='{abs_fbx}', path_mode='COPY', embed_textures=True)",
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(script_lines))
            tmp_script_path = tmp.name

        try:
            proc = subprocess.run(
                [blender_exe, "--background", "--python", tmp_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            return proc.returncode == 0 and os.path.exists(fbx_path) and os.path.getsize(fbx_path) > 0
        except Exception:
            return False
        finally:
            if os.path.exists(tmp_script_path):
                os.remove(tmp_script_path)

    @staticmethod
    def _export_native_fbx_74(src_path: str, fbx_path: str, texture_png_path: str = None) -> str:
        mesh = trimesh.load(src_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("[-] Cannot export empty mesh to FBX.")

        centroid = vertices.mean(axis=0)
        vertices[:, 0] -= centroid[0]
        vertices[:, 2] -= centroid[2]
        vertices[:, 1] -= vertices[:, 1].min()

        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(vertices):
            mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) == len(vertices):
            uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
        else:
            min_b = vertices.min(axis=0)
            span_b = np.maximum(vertices.max(axis=0) - min_b, 1e-5)
            uvs = np.column_stack([
                (vertices[:, 0] - min_b[0]) / span_b[0],
                (vertices[:, 2] - min_b[2]) / span_b[2],
            ]).astype(np.float32)

        fbx_faces = faces.copy()
        fbx_faces[:, 2] = ~fbx_faces[:, 2]

        verts_flat = vertices.ravel()
        norms_flat = normals.ravel()
        uvs_flat = uvs.ravel()
        faces_flat = fbx_faces.ravel()

        verts_str = ",".join(np.char.mod("%.5f", verts_flat))
        norms_str = ",".join(np.char.mod("%.4f", norms_flat))
        uvs_str = ",".join(np.char.mod("%.5f", uvs_flat))
        faces_str = ",".join(np.char.mod("%d", faces_flat))

        if not texture_png_path:
            candidate_tex = os.path.join(os.path.dirname(fbx_path), "final_textured_model_4k_atlas.png")
            texture_png_path = candidate_tex if os.path.exists(candidate_tex) else "final_textured_model_4k_atlas.png"

        rel_tex_name = os.path.basename(texture_png_path)
        abs_tex_name = os.path.abspath(texture_png_path).replace("\\", "/")

        lines = [
            "; FBX 7.4.0 project file",
            "FBXHeaderExtension:  {",
            "    FBXHeaderVersion: 1003",
            "    FBXVersion: 7400",
            '    Creator: "AETHER3D Photogrammetry Pipeline"',
            "}",
            "Definitions:  {",
            "    Version: 100",
            "    Count: 4",
            '    ObjectType: "Model" { Count: 1 }',
            '    ObjectType: "Geometry" { Count: 1 }',
            '    ObjectType: "Material" { Count: 1 }',
            '    ObjectType: "Texture" { Count: 1 }',
            "}",
            "Objects:  {",
            '    Geometry: 100001, "Geometry::AeroTerrainMesh", "Mesh" {',
            f"        Vertices: *{len(verts_flat)} {{",
            f"            a: {verts_str}",
            "        }",
            f"        PolygonVertexIndex: *{len(faces_flat)} {{",
            f"            a: {faces_str}",
            "        }",
            "        GeometryVersion: 124",
            "        LayerElementNormal: 0 {",
            "            Version: 101",
            '            MappingInformationType: "ByVertice"',
            '            ReferenceInformationType: "Direct"',
            f"            Normals: *{len(norms_flat)} {{",
            f"                a: {norms_str}",
            "            }",
            "        }",
            "        LayerElementUV: 0 {",
            "            Version: 101",
            '            MappingInformationType: "ByVertice"',
            '            ReferenceInformationType: "Direct"',
            f"            UV: *{len(uvs_flat)} {{",
            f"                a: {uvs_str}",
            "            }",
            "        }",
            "    }",
            '    Model: 200001, "Model::AeroTerrain_3D", "Mesh" {',
            "        Version: 232",
            "    }",
            '    Material: 300001, "Material::AeroTerrain_4K_PBR", "" {',
            "        Version: 102",
            '        ShadingModel: "phong"',
            "    }",
            '    Texture: 400001, "Texture::Albedo_4K_Atlas", "" {',
            '        Type: "TextureVideoClip"',
            "        Version: 202",
            f'        FileName: "{abs_tex_name}"',
            f'        RelativeFilename: "{rel_tex_name}"',
            "    }",
            "}",
            "Connections:  {",
            '    C: "OO",200001,0',
            '    C: "OO",100001,200001',
            '    C: "OO",300001,200001',
            '    C: "OP",400001,300001, "DiffuseColor"',
            "}",
        ]
        os.makedirs(os.path.dirname(fbx_path), exist_ok=True)
        with open(fbx_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return fbx_path

    @classmethod
    def export_to_fbx(
        cls,
        glb_path: str = "data/outputs/mesh/final_model.glb",
        obj_path: str = "data/outputs/mesh/final_textured_model.obj",
        fallback_obj_path: str = "data/outputs/mesh/surface_mesh_clean.obj",
        output_fbx_path: str = "data/outputs/mesh/final_model.fbx",
    ) -> str:
        src_path = None
        for candidate in [glb_path, obj_path, fallback_obj_path]:
            if candidate and os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                src_path = candidate
                break

        if not src_path:
            raise FileNotFoundError("[-] No source mesh found to convert to FBX.")

        os.makedirs(os.path.dirname(output_fbx_path), exist_ok=True)
        blender_exe = cls._find_blender_executable()
        if blender_exe and cls._export_via_headless_blender(blender_exe, src_path, output_fbx_path):
            print(f"[+] Exported Binary FBX via Blender: {output_fbx_path}")
            return output_fbx_path

        cls._export_native_fbx_74(src_path, output_fbx_path)
        print(f"[+] Exported Autodesk FBX 7.4: {output_fbx_path}")
        return output_fbx_path
