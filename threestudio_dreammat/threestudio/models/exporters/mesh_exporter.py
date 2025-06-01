from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import os

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj-mtl"  # in ['obj-mtl', 'obj'], TODO: fbx
        save_name: str = "model"
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 2048
        texture_format: str = "jpg"
        xatlas_chart_options: dict = field(default_factory=dict)
        xatlas_pack_options: dict = field(default_factory=dict)
        context_type: str = "cuda"

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)

    def __call__(self) -> List[ExporterOutput]:
        meshes_dict: Dict[str, Mesh] = self.geometry.isosurface()
        all_outputs: List[ExporterOutput] = []

        if not isinstance(meshes_dict, dict):
            # Fallback for single mesh for compatibility if geometry returns a single mesh
            meshes_dict = {"model": meshes_dict} 

        for mesh_name, mesh_object in meshes_dict.items():
            if not isinstance(mesh_object, Mesh):
                threestudio.warn(f"Skipping export for {mesh_name} as it is not a valid Mesh object, but {type(mesh_object)}.")
                continue
            
            # Construct path with subdirectory for the mesh type
            current_save_path_prefix = f"{mesh_name}/{self.cfg.save_name}"

            if self.cfg.fmt == "obj-mtl":
                all_outputs.extend(self.export_obj_with_mtl(mesh_object, current_save_path_prefix))
            elif self.cfg.fmt == "obj":
                all_outputs.extend(self.export_obj(mesh_object, current_save_path_prefix))
            else:
                raise ValueError(f"Unsupported mesh export format: {self.cfg.fmt}")
        return all_outputs

    def export_obj_with_mtl(self, mesh: Mesh, save_name_prefix: str) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,  # Base Color
            "map_Ks": None,  # Specular
            "map_Bump": None,  # Normal
            # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
            "map_Pm": None,  # Metallic
            "map_Pr": None,  # Roughness
            "map_format": self.cfg.texture_format,
            # Pass the prefix to be included in texture file paths within the MTL file
            "mtl_path_prefix": os.path.dirname(save_name_prefix) 
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4, mesh.t_tex_idx, (self.cfg.texture_size, self.cfg.texture_size)
            )

            hole_mask = ~(rast[:, :, 3] > 0)

            def uv_padding(image):
                uv_padding_size = self.cfg.xatlas_pack_options.get("padding", 2)
                inpaint_image = (
                    cv2.inpaint(
                        (image.detach().cpu().numpy() * 255).astype(np.uint8),
                        (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                        uv_padding_size,
                        cv2.INPAINT_TELEA,
                    )
                    / 255.0
                )
                return torch.from_numpy(inpaint_image).to(image)

            # Interpolate world space position
            gb_pos, _ = self.ctx.interpolate_one(
                mesh.v_pos, rast[None, ...], mesh.t_pos_idx
            )
            gb_pos = gb_pos[0]

            # Sample out textures from MLP
            geo_out = self.geometry.export(points=gb_pos)
            mat_out = self.material.export(points=gb_pos, **geo_out)

            threestudio.info(
                "Perform UV padding on texture maps to avoid seams, may take a while ..."
            )

            if "albedo" in mat_out:
                params["map_Kd"] = uv_padding(mat_out["albedo"])
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            if "metallic" in mat_out:
                params["map_Pm"] = uv_padding(mat_out["metallic"])
            if "roughness" in mat_out:
                params["map_Pr"] = uv_padding(mat_out["roughness"])
            if "bump" in mat_out:
                params["map_Bump"] = uv_padding(mat_out["bump"])
            # TODO: map_Ks
        return [
            ExporterOutput(
                save_name=f"{save_name_prefix}.obj", save_type="obj", params=params
            )
        ]

    def export_obj(self, mesh: Mesh, save_name_prefix: str) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": False,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,  # Base Color
            "map_Ks": None,  # Specular
            "map_Bump": None,  # Normal
            # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
            "map_Pm": None,  # Metallic
            "map_Pr": None,  # Roughness
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            geo_out = self.geometry.export(points=mesh.v_pos)
            mat_out = self.material.export(points=mesh.v_pos, **geo_out)

            if "albedo" in mat_out:
                mesh.set_vertex_color(mat_out["albedo"])
                params["save_vertex_color"] = True
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, not saving vertex color"
                )

        return [
            ExporterOutput(
                save_name=f"{save_name_prefix}.obj", save_type="obj", params=params
            )
        ]
