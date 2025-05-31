import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from threestudio.models.renderers.raytracing_renderer import RayTracer

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *
import subprocess,os
import pickle
import cv2
import numpy as np

def fovy_to_focal(fovy, sensor_height):
    return sensor_height / (2 * math.tan(fovy / 2))

target_mesh_path = "load/shapes/objs/target.obj"
other_mesh_path = "load/shapes/objs/other.obj"
target_mesh = None
other_mesh = None

@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    fix_view_num: int = 128
    fix_env_num: int = 5
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    use_fix_views: bool = True
    blender_generate: bool = False

class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb = (
            torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        env_id: Int[Tensor,"B"]=(torch.rand(self.batch_size)*self.cfg.fix_env_num).floor().int()
        #env_id[0]=3
        return {
            "env_id":env_id,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            #'k':Ks,
            "c2w": c2w,
            "w2c": w2c,
            "light_positions": None,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
        }

def xfm_vectors(vectors, matrix):
    out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out

def saveimg(img,path):
    from PIL import Image
    # Ensure img is a numpy array before creating PIL Image
    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()
    # Handle potential different shapes (e.g., (H, W, 1) or (H, W, 3))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1) # Remove last dimension if it's 1
    # Convert to uint8 (0-255)
    img = (img * 255).astype(np.uint8)
    # Ensure correct shape for PIL Image.fromarray
    if img.ndim == 2:
         img = np.stack([img]*3, axis=-1) # Convert grayscale to RGB if needed
    img = Image.fromarray(img)
    img.save(path)

def loadrgb(imgpath,dim):
    img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    return img

def loaddepth(imgpath,dim):
    depth = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH)/1000
    depth = cv2.resize(depth, dim, interpolation = cv2.INTER_NEAREST)
    object_mask = depth>0
        
    if object_mask.sum()<=0:
        print(imgpath)
        return depth[...,None]

    min_val=0.3

    depth_inv = 1. / (depth + 1e-6)
    
    depth_max = depth_inv[object_mask].max()
    depth_min = depth_inv[object_mask].min()
                    
    depth[object_mask] = (1 - min_val) *(depth_inv[object_mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val
    return depth[...,None]

class FixCameraIterableDataset(IterableDataset, Updateable):

    def render_oneview_gt(self, view_id):
        elevation_deg: Float[Tensor, "B"] = self.elevation_degs[view_id]
        elevation: Float[Tensor, "B"] = elevation_deg * math.pi / 180

        azimuth_deg: Float[Tensor, "B"] = self.azimuth_degs[view_id]
        azimuth: Float[Tensor, "B"] = azimuth_deg * math.pi / 180

        camera_distances: Float[Tensor, "B"] = self.fix_camera_distances[view_id]

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb = self.camera_perturbs[view_id,...]
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb = self.center_perturbs[view_id,...]
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb = self.up_perturbs[view_id,...]
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = self.fovy_degs[view_id]
        fovy = fovy_deg * math.pi / 180

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        # Get target mesh directly from the mesh dictionary

        for name in self.mesh.keys():
            mesh = self.mesh[name]
            ray_tracer = RayTracer(mesh.v_pos, mesh.t_pos_idx)
            inters, normals, depth = ray_tracer.trace(rays_o.reshape(-1,3), rays_d.reshape(-1,3))
            normals = F.normalize(normals, dim=-1)
            miss_mask = depth >= 10
            hit_mask = ~miss_mask

            normal_view  = xfm_vectors(normals[hit_mask].view(view_id.shape[0], normals[hit_mask].shape[0], normals[hit_mask].shape[1]), w2c.to(normals.device)).view(*normals[hit_mask].shape)
            normal_view = F.normalize(normal_view)
            normal_controlnet=0.5*(normal_view+1)
            normal_controlnet[..., 0]=1.0-normal_controlnet[..., 0] # Flip the sign on the x-axis to match bae system
            normals[hit_mask]=normal_controlnet

            min_val=0.3
            depth_inv = 1. / (depth + 1e-6)
            depth_max = depth_inv[hit_mask].max()
            depth_min = depth_inv[hit_mask].min()
            depth[hit_mask] = (1 - min_val) *(depth_inv[hit_mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val
            depth[~hit_mask]=0.0

            hit_mask=hit_mask.reshape((self.height,self.width,1))
            hit_mask=hit_mask.repeat(1,1,3).float()
            depth=depth.reshape((self.height,self.width,1))
            depth=depth.repeat(1,1,3)
            normals=normals.reshape((self.height,self.width,3))
            saveimg(normals,self.temp_image_save_dir+f'/submesh_{name}' +'/gt/normal'+str(view_id[0])+'.png')

        return normals, depth, hit_mask

    def render_fixview_imgs(self):
        envmap_dir = "load/lights/envmap"

        # Initialize sub_meshes dictionary
        sub_meshes = {
            'target': {
                'depth_dir': os.path.join(self.temp_image_save_dir, 'submesh_target/depth'),
                'normal_dir': os.path.join(self.temp_image_save_dir, 'submesh_target/normal'),
                'light_dir': os.path.join(self.temp_image_save_dir, 'submesh_target/light')
            },
            'other': {
                'depth_dir': os.path.join(self.temp_image_save_dir, 'submesh_other/depth'),
                'normal_dir': os.path.join(self.temp_image_save_dir, 'submesh_other/normal'),
                'light_dir': os.path.join(self.temp_image_save_dir, 'submesh_other/light')
            }
        }

        elevation_deg = self.elevation_degs
        elevation = elevation_deg * math.pi / 180
        azimuth_deg = self.azimuth_degs
        azimuth = azimuth_deg * math.pi / 180
        camera_distances = self.fix_camera_distances
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],dim=-1,
        )
        
        camera_perturb = self.camera_perturbs
        camera_positions = camera_positions + camera_perturb
        center_perturb = self.center_perturbs
        center = torch.zeros_like(camera_positions)
        center = center + center_perturb
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1).expand(128, -1)
        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1,)
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0
        fovy_deg = self.fovy_degs
        fovy = fovy_deg * math.pi / 180
        focal_length = 0.5 * self.height / torch.tan(0.5 * fovy)
        
        if self.cfg.blender_generate:
            # Get meshes from DreamMatMesh
            
            # Process each mesh type
            for name, mesh in self.mesh.items():
                sub_mesh_data = {}
                sub_mesh_data['v_pos'] = mesh.v_pos.cpu().numpy()
                sub_mesh_data['t_pos_idx'] = mesh.t_pos_idx.cpu().numpy()
                sub_mesh_data['width'] = self.width
                sub_mesh_data['height'] = self.height
                sub_mesh_data['focal_length'] = focal_length.cpu().numpy()
                sub_mesh_data['c2w'] = c2w.cpu().numpy()
                    
                sub_mesh_dir = os.path.join(self.temp_image_save_dir, f'submesh_{name}')
                os.makedirs('temp', exist_ok=True)
                os.makedirs(sub_mesh_dir, exist_ok=True)
                pkl_path = f'temp/render_fixview_temp_{name}.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(sub_mesh_data, f)
                    
                # Render for this sub-mesh
                cmd = f'blender -b -P ./threestudio/data/blender_script_fixview.py -- --param_dir {pkl_path} --env_dir {envmap_dir} --output_dir {sub_mesh_dir} --num_images {self.cfg.fix_view_num}'
                print(f"Rendering sub-mesh for {name}...")
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
                print(f"Rendering done for {name}")
            
        # Load and combine rendered images
        self.depths = {
            'target': torch.zeros((128, self.height, self.width, 1)),
            'other': torch.zeros((128, self.height, self.width, 1))
        }
        self.normals = {
            'target': torch.ones((128, self.height, self.width, 3)),
            'other': torch.ones((128, self.height, self.width, 3))
        }
        self.lightmaps = {
            'target': torch.zeros((128, 5, self.height, self.width, 18)),
            'other': torch.zeros((128, 5, self.height, self.width, 18))
        }
            
        dim = (self.width, self.height)
        for view_idx in range(self.cfg.fix_view_num):
            # Combine depth maps
            for name, paths in sub_meshes.items():
                depth_path = os.path.join(paths['depth_dir'], f"{view_idx:03d}.png")
                if os.path.exists(depth_path):
                    depth = loaddepth(depth_path, dim)
                    if depth is not None : self.depths[name][view_idx] = torch.from_numpy(depth)
                
            # Combine normal maps
            for name, paths in sub_meshes.items():
                normal_path = os.path.join(paths['normal_dir'], f"{view_idx:03d}.png")
                if os.path.exists(normal_path):
                    normal = loadrgb(normal_path, dim)
                    object_mask = np.any(normal != 1.0, axis=-1)
                    self.normals[name][view_idx] = torch.from_numpy(normal)
                
            # Combine light maps
            for env_idx in range(1, 6):
                for name, paths in sub_meshes.items():
                    light_path_m0r0 = os.path.join(paths['light_dir'], f"{view_idx:03d}_m0.0r0.0_env{env_idx}.png")
                    light_path_m0rhalf = os.path.join(paths['light_dir'], f"{view_idx:03d}_m0.0r0.5_env{env_idx}.png")
                    light_path_m0r1 = os.path.join(paths['light_dir'], f"{view_idx:03d}_m0.0r1.0_env{env_idx}.png")
                    light_path_m1r0 = os.path.join(paths['light_dir'], f"{view_idx:03d}_m1.0r0.0_env{env_idx}.png")
                    light_path_m1rhalf = os.path.join(paths['light_dir'], f"{view_idx:03d}_m1.0r0.5_env{env_idx}.png")
                    light_path_m1r1 = os.path.join(paths['light_dir'], f"{view_idx:03d}_m1.0r1.0_env{env_idx}.png")
                        
                    if all(os.path.exists(p) for p in [
                        light_path_m0r0, light_path_m0rhalf, light_path_m0r1, 
                        light_path_m1r0, light_path_m1rhalf, light_path_m1r1]
                    ):
                        light_m0r0 = loadrgb(light_path_m0r0, dim)
                        light_m0rhalf = loadrgb(light_path_m0rhalf, dim)
                        light_m0r1 = loadrgb(light_path_m0r1, dim)
                        light_m1r0 = loadrgb(light_path_m1r0, dim)
                        light_m1rhalf = loadrgb(light_path_m1rhalf, dim)
                        light_m1r1 = loadrgb(light_path_m1r1, dim)
                            
                        light_combined = np.concatenate([
                            light_m0r0, light_m0rhalf, light_m0r1,
                            light_m1r0, light_m1rhalf, light_m1r1
                        ], axis=-1)
                            
                        self.lightmaps[name][view_idx, env_idx-1] = torch.from_numpy(light_combined)

    def set_fix_elevs(self) -> None:
        elevation_degs1 = (
                torch.rand(int(self.cfg.fix_view_num/2))
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
        elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
        elevation = torch.asin(
                2
                * (
                    torch.rand(int(self.cfg.fix_view_num/2))
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
        elevation_degs2 = elevation / math.pi * 180.0
        self.elevation_degs=torch.cat((elevation_degs1,elevation_degs2))
    
    def set_fix_azims(self) ->None:
        self.azimuth_degs = (
                torch.rand(self.cfg.fix_view_num) + torch.arange(self.cfg.fix_view_num)
            ) / self.cfg.fix_view_num * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]

    def set_fix_camera_distance(self) -> None:
        self.fix_camera_distances= (
            torch.rand(self.cfg.fix_view_num)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )
    
    def set_fix_camera_perturb(self) -> None:
        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        self.camera_perturbs = (
            torch.rand(self.cfg.fix_view_num, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
    
    def set_fix_center_perturb(self) ->None:
        self.center_perturbs = (
            torch.randn(self.cfg.fix_view_num, 3) * self.cfg.center_perturb
        )
    
    def set_fix_up_perturb(self) -> None:
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        self.up_perturbs = (
            torch.randn(self.cfg.fix_view_num, 3) * self.cfg.up_perturb
        )

    def set_fix_fovy(self) -> None:
        self.fovy_degs = (
            torch.rand(self.cfg.fix_view_num) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )

    def __init__(self, cfg: Any, mesh,prerender_dir) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.mesh = mesh
        self.temp_image_save_dir=prerender_dir
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

        # Initialize depths, normals, and lightmaps
        self.depths = {
            'target': torch.zeros((self.cfg.fix_view_num, self.height, self.width, 1)),
            'other': torch.zeros((self.cfg.fix_view_num, self.height, self.width, 1))
        }
        self.normals = {
            'target': torch.ones((self.cfg.fix_view_num, self.height, self.width, 3)),
            'other': torch.ones((self.cfg.fix_view_num, self.height, self.width, 3))
        }
        self.lightmaps = {
            'target': torch.zeros((self.cfg.fix_view_num, self.cfg.fix_env_num, self.height, self.width, 18)),
            'other': torch.zeros((self.cfg.fix_view_num, self.cfg.fix_env_num, self.height, self.width, 18))
        }

        self.set_fix_elevs()
        self.set_fix_azims()
        self.set_fix_camera_distance()
        self.set_fix_camera_perturb()
        self.set_fix_center_perturb()
        self.set_fix_up_perturb()
        self.set_fix_fovy()

        os.makedirs(self.temp_image_save_dir+"/submesh_target"+"/gt", exist_ok=True)
        os.makedirs(self.temp_image_save_dir+"/submesh_other"+"/gt", exist_ok=True)
        for i in range(self.cfg.fix_view_num):
            gt_view_id=torch.ones((1))*i 
            self.render_oneview_gt(gt_view_id.long()) 

        self.render_fixview_imgs()

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        #self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # Create separate batch for each submesh type
        batch = {
            'target': {},
            'other': {}
        }

        # Sample view_id for both target and other submeshes
        view_id_target = (torch.rand(self.batch_size)*self.cfg.fix_view_num).floor().long()
        view_id_other = (torch.rand(self.batch_size)*self.cfg.fix_view_num).floor().long()
        
        # Sample env_id for both target and other submeshes
        env_id_target = (torch.rand(self.batch_size)*self.cfg.fix_env_num).floor().long()
        env_id_other = (torch.rand(self.batch_size)*self.cfg.fix_env_num).floor().long()

        # Process target submesh
        elevation_deg_target: Float[Tensor, "B"] = self.elevation_degs[view_id_target]
        elevation_target: Float[Tensor, "B"] = elevation_deg_target * math.pi / 180
        azimuth_deg_target: Float[Tensor, "B"] = self.azimuth_degs[view_id_target]
        azimuth_target: Float[Tensor, "B"] = azimuth_deg_target * math.pi / 180
        camera_distances_target: Float[Tensor, "B"] = self.fix_camera_distances[view_id_target]

        # Process other submesh
        elevation_deg_other: Float[Tensor, "B"] = self.elevation_degs[view_id_other]
        elevation_other: Float[Tensor, "B"] = elevation_deg_other * math.pi / 180
        azimuth_deg_other: Float[Tensor, "B"] = self.azimuth_degs[view_id_other]
        azimuth_other: Float[Tensor, "B"] = azimuth_deg_other * math.pi / 180
        camera_distances_other: Float[Tensor, "B"] = self.fix_camera_distances[view_id_other]

        # Convert spherical coordinates to cartesian coordinates for target submesh
        camera_positions_target = torch.stack(
            [
                camera_distances_target * torch.cos(elevation_target) * torch.cos(azimuth_target),
                camera_distances_target * torch.cos(elevation_target) * torch.sin(azimuth_target),
                camera_distances_target * torch.sin(elevation_target),
            ],
            dim=-1,
        )

        # Convert spherical coordinates to cartesian coordinates for other submesh
        camera_positions_other = torch.stack(
            [
                camera_distances_other * torch.cos(elevation_other) * torch.cos(azimuth_other),
                camera_distances_other * torch.cos(elevation_other) * torch.sin(azimuth_other),
                camera_distances_other * torch.sin(elevation_other),
            ],
            dim=-1,
        )

        # Default scene center at origin for both submeshes
        center_target = torch.zeros_like(camera_positions_target)
        center_other = torch.zeros_like(camera_positions_other)

        # Default camera up direction as +z for both submeshes
        up_target = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1)
        up_other = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1)

        # Apply perturbations for target submesh
        camera_perturb_target = self.camera_perturbs[view_id_target,...]
        camera_positions_target = camera_positions_target + camera_perturb_target
        center_perturb_target = self.center_perturbs[view_id_target,...]
        center_target = center_target + center_perturb_target
        up_perturb_target = self.up_perturbs[view_id_target,...]
        up_target = up_target + up_perturb_target

        # Apply perturbations for other submesh
        camera_perturb_other = self.camera_perturbs[view_id_other,...]
        camera_positions_other = camera_positions_other + camera_perturb_other
        center_perturb_other = self.center_perturbs[view_id_other,...]
        center_other = center_other + center_perturb_other
        up_perturb_other = self.up_perturbs[view_id_other,...]
        up_other = up_other + up_perturb_other

        # Get fovy for both submeshes
        fovy_deg_target: Float[Tensor, "B"] = self.fovy_degs[view_id_target]
        fovy_target = fovy_deg_target * math.pi / 180
        fovy_deg_other: Float[Tensor, "B"] = self.fovy_degs[view_id_other]
        fovy_other = fovy_deg_other * math.pi / 180

        # Calculate camera matrices for target submesh
        lookat_target = F.normalize(center_target - camera_positions_target, dim=-1)
        right_target = F.normalize(torch.cross(lookat_target, up_target), dim=-1)
        up_target = F.normalize(torch.cross(right_target, lookat_target), dim=-1)
        c2w3x4_target = torch.cat(
            [torch.stack([right_target, up_target, -lookat_target], dim=-1), camera_positions_target[:, :, None]],
            dim=-1,
        )
        c2w_target = torch.cat(
            [c2w3x4_target, torch.zeros_like(c2w3x4_target[:, :1])], dim=1
        )
        c2w_target[:, 3, 3] = 1.0

        # Calculate camera matrices for other submesh
        lookat_other = F.normalize(center_other - camera_positions_other, dim=-1)
        right_other = F.normalize(torch.cross(lookat_other, up_other), dim=-1)
        up_other = F.normalize(torch.cross(right_other, lookat_other), dim=-1)
        c2w3x4_other = torch.cat(
            [torch.stack([right_other, up_other, -lookat_other], dim=-1), camera_positions_other[:, :, None]],
            dim=-1,
        )
        c2w_other = torch.cat(
            [c2w3x4_other, torch.zeros_like(c2w3x4_other[:, :1])], dim=1
        )
        c2w_other[:, 3, 3] = 1.0

        # Calculate directions and rays for target submesh
        focal_length_target = 0.5 * self.height / torch.tan(0.5 * fovy_target)
        directions_target = self.directions_unit_focal[None, :, :, :].repeat(self.batch_size, 1, 1, 1)
        directions_target[:, :, :, :2] = directions_target[:, :, :, :2] / focal_length_target[:, None, None, None]
        rays_o_target, rays_d_target = get_rays(directions_target, c2w_target, keepdim=True)
        proj_mtx_target = get_projection_matrix(fovy_target, self.width / self.height, 0.1, 1000.0)
        mvp_mtx_target, w2c_target = get_mvp_matrix(c2w_target, proj_mtx_target)

        # Calculate directions and rays for other submesh
        focal_length_other = 0.5 * self.height / torch.tan(0.5 * fovy_other)
        directions_other = self.directions_unit_focal[None, :, :, :].repeat(self.batch_size, 1, 1, 1)
        directions_other[:, :, :, :2] = directions_other[:, :, :, :2] / focal_length_other[:, None, None, None]
        rays_o_other, rays_d_other = get_rays(directions_other, c2w_other, keepdim=True)
        proj_mtx_other = get_projection_matrix(fovy_other, self.width / self.height, 0.1, 1000.0)
        mvp_mtx_other, w2c_other = get_mvp_matrix(c2w_other, proj_mtx_other)

        # Get condition maps for both submeshes
        target_depth = self.depths['target'][view_id_target,...]
        target_normal = self.normals['target'][view_id_target,...]
        target_light = self.lightmaps['target'][view_id_target,env_id_target,...]

        other_depth = self.depths['other'][view_id_other,...]
        other_normal = self.normals['other'][view_id_other,...]
        other_light = self.lightmaps['other'][view_id_other,env_id_other,...]

        # Combine condition maps for both submeshes
        target_condition_map = torch.cat((target_depth, target_normal, target_light), -1)
        other_condition_map = torch.cat((other_depth, other_normal, other_light), -1)

        # Populate target batch
        batch['target'] = {
            "env_id": env_id_target,
            "rays_o": rays_o_target,
            "rays_d": rays_d_target,
            "mvp_mtx": mvp_mtx_target,
            "camera_positions": camera_positions_target,
            "c2w": c2w_target,
            "w2c": w2c_target,
            "light_positions": None,
            "elevation": elevation_deg_target,
            "azimuth": azimuth_deg_target,
            "camera_distances": camera_distances_target,
            "condition_map": target_condition_map,
            "height": self.height,
            "width": self.width,
        }

        # Populate other batch
        batch['other'] = {
            "env_id": env_id_other,
            "rays_o": rays_o_other,
            "rays_d": rays_d_other,
            "mvp_mtx": mvp_mtx_other,
            "camera_positions": camera_positions_other,
            "c2w": c2w_other,
            "w2c": w2c_other,
            "light_positions": None,
            "elevation": elevation_deg_other,
            "azimuth": azimuth_deg_other,
            "camera_distances": camera_distances_other,
            "condition_map": other_condition_map,
            "height": self.height,
            "width": self.width,
        }

        return batch

class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str, mesh, prerender_dir) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split
        self.mesh = mesh
        self.prerender_dir = prerender_dir

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        light_positions = camera_positions

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.w2c = w2c
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances
        self.fovy = fovy
        self.fovy_deg = fovy_deg

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "env_id": 4,
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "w2c": self.w2c[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch

@register("random-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, mesh, prerender_dir,cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)
        self.mesh=mesh
        self.prerender_dir=prerender_dir
        os.makedirs(self.prerender_dir, exist_ok=True)

    def setup(self, stage=None) -> None:
        target_mesh = self.mesh['target']
        other_mesh = self.mesh['other']

        if stage in [None, "fit"]:
            if self.cfg.use_fix_views:
                self.train_dataset = FixCameraIterableDataset(self.cfg,self.mesh,self.prerender_dir)
            else:
                self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val", target_mesh, self.prerender_dir)
            self.val_dataset = RandomCameraDataset(self.cfg, "val", other_mesh, self.prerender_dir)
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test", target_mesh, self.prerender_dir)
            self.test_dataset = RandomCameraDataset(self.cfg, "test", other_mesh, self.prerender_dir)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )