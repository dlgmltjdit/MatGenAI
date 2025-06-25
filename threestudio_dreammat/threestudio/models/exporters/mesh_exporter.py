from dataclasses import dataclass, field
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.ops import get_mvp_matrix, get_projection_matrix
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj-mtl"
        save_name: str = "model"
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 2048
        texture_format: str = "jpg"
        xatlas_chart_options: dict = field(default_factory=dict)
        xatlas_pack_options: dict = field(default_factory=dict)
        context_type: str = "cuda"
        render_multiview: bool = True
        render_camera_distance: float = 4.0
        render_num_views: int = 12
        render_elevation_deg: float = 15.0
        render_fovy_deg: float = 60.0

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
            meshes_dict = {"model": meshes_dict}

        # Part 1: Export individual meshes in their internal (transformed) coordinate system
        for mesh_name, mesh_object in meshes_dict.items():
            if not isinstance(mesh_object, Mesh):
                threestudio.warn(
                    f"Skipping individual export for {mesh_name} as it is not a valid Mesh object."
                )
                continue

            save_path = os.path.join(mesh_name, self.cfg.save_name)
            all_outputs.extend(self.export_single_mesh(mesh_object, save_path))

        # Part 2: Render a composite of all meshes in their original coordinate system
        if self.cfg.render_multiview and len(meshes_dict) > 1:
            all_outputs.extend(
                self.render_composite_multiview(
                    meshes_dict, self.material, self.background
                )
            )

        return all_outputs

    def export_single_mesh(
        self, mesh: Mesh, save_name: str
    ) -> List[ExporterOutput]:
        """
        Exports a single mesh with its own texture.
        The mesh is assumed to be in the transformed coordinate system used for training.
        """
        params = {
            "mesh": mesh,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_Pm": None,
            "map_Pr": None,
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info(f"Exporting texture for {save_name}...")
            assert self.cfg.save_uv, "save_uv must be True to save texture."

            uv_clip = mesh.v_tex * 2.0 - 1.0
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )

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

            gb_pos, _ = self.ctx.interpolate_one(
                mesh.v_pos, rast[None, ...], mesh.t_pos_idx
            )
            gb_pos = gb_pos[0]

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

        return [
            ExporterOutput(
                save_name=f"{save_name}.obj", save_type="obj", params=params
            )
        ]

    def render_composite_multiview(
        self,
        meshes_dict: Dict[str, Mesh],
        material: BaseMaterial,
        background: BaseBackground,
    ) -> List[ExporterOutput]:
        """
        Renders a composite of multiple meshes from different viewpoints.
        Each mesh is placed in its original coordinate system and shaded with lighting.
        """
        threestudio.info(f"Rendering {self.cfg.render_num_views} composite views...")
        height = width = self.cfg.texture_size
        num_views = self.cfg.render_num_views
        
        # Use camera settings from config, matching the test dataset defaults
        camera_distance = self.cfg.render_camera_distance
        elevation_deg = self.cfg.render_elevation_deg
        fovy_deg = self.cfg.render_fovy_deg
        azimuths_deg = torch.linspace(0, 360.0, num_views + 1)[:-1]

        # Initialize final image buffer
        final_images = []

        # Loop and render each view sequentially
        for i in range(num_views):
            threestudio.info(f"Rendering view {i+1}/{num_views}...")
            
            # --- Per-view camera setup ---
            azimuth_rad = torch.deg2rad(azimuths_deg[i]).to(self.device)
            elevation_rad = torch.deg2rad(torch.tensor(elevation_deg, device=self.device))
            
            x = camera_distance * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
            y = camera_distance * torch.sin(elevation_rad)
            z = camera_distance * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
            camera_position = torch.stack([x, y, z]).view(1, 3)

            center = torch.zeros_like(camera_position)
            world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).view(1, 3)

            lookat = F.normalize(center - camera_position, dim=-1)
            right = F.normalize(torch.cross(lookat, world_up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4 = torch.cat(
                [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
                dim=-1,
            )
            c2w_matrix = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
            c2w_matrix[:, 3, 3] = 1.0

            fovy = torch.deg2rad(torch.tensor(fovy_deg, dtype=torch.float32, device=self.device))
            proj_mtx = get_projection_matrix(fovy.view(1), width / height, 0.1, 1000.0)
            mvp_mtx, _ = get_mvp_matrix(c2w_matrix, proj_mtx.to(self.device))
            
            env_id = torch.tensor([4], dtype=torch.long, device=self.device)
            
            # --- Per-view composition ---
            final_rgb_view = torch.zeros(height, width, 3, device=self.device)
            depth_buffer_view = torch.full(
                (height, width, 1), float("inf"), device=self.device
            )
            final_rgb_view += background.env_color.view(1, 1, 3)

            for mesh in meshes_dict.values():
                v_pos_original = mesh.extras.get("original_v_pos", mesh.v_pos)
                v_nrm_original = mesh.extras.get("original_v_nrm", mesh.v_nrm)
                v_pos_transformed = mesh.v_pos

                v_pos_clip = self.ctx.vertex_transform(v_pos_original, mvp_mtx)
                rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
                rast = rast.squeeze(0) # Remove batch dimension

                mask = rast[..., 3:4] > 0
                depth = rast[..., 2:3]
                
                update_mask = (mask) & (depth < depth_buffer_view)
                if not torch.any(update_mask):
                    continue

                gb_pos, _ = self.ctx.interpolate(v_pos_transformed, rast.unsqueeze(0), mesh.t_pos_idx)
                gb_nrm, _ = self.ctx.interpolate(v_nrm_original, rast.unsqueeze(0), mesh.t_pos_idx)
                
                update_mask_flat = update_mask.squeeze(-1)
                pts_flat = gb_pos.squeeze(0)[update_mask_flat]
                normals_flat = F.normalize(gb_nrm.squeeze(0)[update_mask_flat], dim=-1)

                cam_pos_expanded = camera_position.view(1, 1, 3).expand(height, width, -1)
                viewdirs_flat = F.normalize(cam_pos_expanded[update_mask_flat] - pts_flat, dim=-1)

                geo_out = self.geometry.export(points=pts_flat)
                features_flat = geo_out['features']
                
                # Explicitly get all material properties using the export method
                mat_out = material.export(points=pts_flat, **geo_out)
                
                shade_outputs, _ = material(
                    pts=pts_flat, 
                    features=features_flat, 
                    features_jitter=features_flat,
                    viewdirs=viewdirs_flat, 
                    normals=normals_flat, 
                    env_id=env_id,
                    **mat_out # Pass albedo, metallic, roughness etc.
                )
                color_flat = shade_outputs['color']

                final_rgb_view[update_mask_flat] = color_flat
                depth_buffer_view[update_mask] = depth[update_mask]

            final_images.append(final_rgb_view)

        outputs = []
        for i, img_tensor in enumerate(final_images):
            img = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            outputs.append(
                ExporterOutput(
                    save_name=f"renders/img_{i:03d}.png",
                    save_type="image",
                    params={"img": img},
                )
            )
        
        threestudio.info("Finished composite rendering.")
        return outputs
