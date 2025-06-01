from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.ops import (
            get_mvp_matrix,
            get_projection_matrix,
            get_ray_directions,
            get_rays,
        )
import math
import random
from threestudio.utils.ops import get_activation
import os
from threestudio.systems.utils import parse_optimizer, parse_scheduler

@threestudio.register("dreammat-system")
class DreamMat(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture:bool=True
        latent_steps: int = 1000
        save_train_image: bool = True
        save_train_image_iter: int = 1
        init_step: int = 0
        init_width:int=512
        init_height:int=512
        test_background_white: Optional[bool] = False
        target_prompt: Optional[str] = None

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        # Manual optimization 모드로 설정
        self.automatic_optimization = False

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def configure_optimizers(self):
        optimizer_target = parse_optimizer(self.cfg.optimizer, self)
        optimizer_other = parse_optimizer(self.cfg.optimizer, self)

        optimizers = [optimizer_target, optimizer_other]

        if self.cfg.scheduler is not None:
            # Create scheduler configs for each optimizer
            # parse_scheduler returns a dict like {'scheduler': sched_instance, 'interval': 'step', ...}
            scheduler_target_config = parse_scheduler(self.cfg.scheduler, optimizer_target)
            scheduler_other_config = parse_scheduler(self.cfg.scheduler, optimizer_other)
            return optimizers, [scheduler_target_config, scheduler_other_config]
        else:
            return optimizers

    def training_step(self, batch, batch_idx):
        opt_target, opt_other = self.optimizers()

        prompt_outputs = self.prompt_processor()
        
        # Initialize losses to be returned
        target_loss_val = torch.tensor(0.0, device=self.device)
        other_loss_val = torch.tensor(0.0, device=self.device)

        # Process target submesh
        submesh_type_target = "target"
        submesh_batch_target = batch[submesh_type_target]
        prompt_utils_target = prompt_outputs[submesh_type_target]
        
        opt_target.zero_grad()
        out_target = self(submesh_batch_target)
        guidance_inp_target = out_target[submesh_type_target]["comp_rgb"]
        submesh_batch_target['cond_normal'] = out_target[submesh_type_target].get('comp_normal', None)
        submesh_batch_target['cond_depth'] = out_target[submesh_type_target].get('comp_depth', None)

        guidance_out_target = self.guidance(
            guidance_inp_target, prompt_utils_target, **submesh_batch_target, rgb_as_latents=False,
        )
        current_target_loss = 0.0
        for name, value in guidance_out_target.items():
            self.log(f"train/{submesh_type_target}_{name}", value, prog_bar=True, logger=True, sync_dist=True)
            if name.startswith("loss_"):
                current_target_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        for name, value in out_target[submesh_type_target].items():
            if name.startswith("loss_"):
                self.log(f"train/{submesh_type_target}_{name}", value, prog_bar=True, logger=True, sync_dist=True)
                current_target_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        self.manual_backward(current_target_loss, retain_graph=True) # Retain graph for the other optimizer
        opt_target.step()
        target_loss_val = current_target_loss.detach() # Detach for returning/logging

        # Process other submesh
        submesh_type_other = "other"
        submesh_batch_other = batch[submesh_type_other]
        prompt_utils_other = prompt_outputs[submesh_type_other]
        
        opt_other.zero_grad()
        out_other = self(submesh_batch_other)
        guidance_inp_other = out_other[submesh_type_other]["comp_rgb"]
        submesh_batch_other['cond_normal'] = out_other[submesh_type_other].get('comp_normal', None)
        submesh_batch_other['cond_depth'] = out_other[submesh_type_other].get('comp_depth', None)

        guidance_out_other = self.guidance(
            guidance_inp_other, prompt_utils_other, **submesh_batch_other, rgb_as_latents=False,
        )
        current_other_loss = 0.0
        for name, value in guidance_out_other.items():
            self.log(f"train/{submesh_type_other}_{name}", value, prog_bar=True, logger=True, sync_dist=True)
            if name.startswith("loss_"):
                current_other_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        for name, value in out_other[submesh_type_other].items():
            if name.startswith("loss_"):
                self.log(f"train/{submesh_type_other}_{name}", value, prog_bar=True, logger=True, sync_dist=True)
                current_other_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        self.manual_backward(current_other_loss)
        opt_other.step()
        other_loss_val = current_other_loss.detach()

        # Log combined or individual losses as needed
        self.log("train/target_loss", target_loss_val, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/other_loss", other_loss_val, prog_bar=True, logger=True, sync_dist=True)

        # Save training images (moved outside the optimizer-specific blocks for clarity)
        # This will save images based on the latest state after both optimizer steps
        if self.cfg.save_train_image and self.true_global_step % self.cfg.save_train_image_iter == 0:
            for submesh_type, out_data in [(submesh_type_target, out_target), (submesh_type_other, out_other)]:
                submesh_batch_for_saving = batch[submesh_type] # Get the original submesh_batch
                train_images_row = [
                    {
                        "type": "rgb",
                        "img": out_data[submesh_type]["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out_data[submesh_type]["specular_light"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out_data[submesh_type]["diffuse_light"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out_data[submesh_type]["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out_data[submesh_type]["comp_depth"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out_data[submesh_type]["albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out_data[submesh_type]["metalness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out_data[submesh_type]["roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                train_conditions_row = [
                    {
                        "type": "grayscale",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 1:4],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 4:7],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 7:10],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 10:13],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 13:16],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 16:19],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch_for_saving["condition_map"][0, :, :, 19:22],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                self.save_image_grid(
                    f"train/it{self.true_global_step}_{submesh_type}.png",
                    imgs=[train_images_row, train_conditions_row],
                    name=f"train_step_{submesh_type}",
                    step=self.true_global_step,
                )

        # PyTorch Lightning doesn't strictly need a return with manual optimization, 
        # but returning losses can be useful for callbacks or if you change optimization strategy later.
        # However, to avoid issues with how Lightning handles dicts from training_step in manual mode,
        # we will not return anything here, as losses are logged directly.
        # return {
        #     "loss": target_loss_val + other_loss_val, # Example: or keep them separate
        #     "target_loss": target_loss_val,
        #     "other_loss": other_loss_val
        # }

    def validation_step(self, batch):
        # Get prompt processor outputs for both submesh types
        prompt_outputs = self.prompt_processor()
        
        for submesh_type in ['target', 'other']:
            # Get prompt processor output for this submesh type
            prompt_utils = prompt_outputs[submesh_type]
            
            out = self(batch)
            srgb = out[submesh_type]["comp_rgb"][0].detach()
            self.save_image_grid(
                f"validate/it{self.true_global_step}-{submesh_type}-{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": srgb,
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if self.cfg.texture
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["specular_light"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["diffuse_light"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["specular_color"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["diffuse_color"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out[submesh_type]["metalness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out[submesh_type]["roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name=f"validation_step_{submesh_type}",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch):
        # Get prompt processor outputs for both submesh types
        prompt_outputs = self.prompt_processor()
        
        for submesh_type in ['target', 'other']:
            # Get prompt processor output for this submesh type
            prompt_utils = prompt_outputs[submesh_type]
            
            out = self(batch)
            srgb = out[submesh_type]["comp_rgb"][0].detach()
            self.save_image_grid(
                f"it{self.true_global_step}-test/view/{submesh_type}/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": srgb,
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if self.cfg.texture
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out[submesh_type]["metalness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out[submesh_type]["roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name=f"test_step_{submesh_type}",
                step=self.true_global_step,
            )
            
            mask = out[submesh_type]["opacity"][0].detach()
            albedo = out[submesh_type]["albedo"][0].detach()
            roughness = out[submesh_type]["roughness"][0].detach().repeat(1,1,3)
            metallic = out[submesh_type]["metalness"][0].detach().repeat(1,1,3)
            
            self.save_img(torch.cat((albedo,mask),2), f"it{self.true_global_step}-test/albedo/{submesh_type}/{batch['index'][0]}.png")
            self.save_img(torch.cat((roughness,mask),2), f"it{self.true_global_step}-test/roughness/{submesh_type}/{batch['index'][0]}.png")
            self.save_img(torch.cat((metallic,mask),2), f"it{self.true_global_step}-test/metallic/{submesh_type}/{batch['index'][0]}.png")
            self.save_img(torch.cat((srgb,mask),2), f"it{self.true_global_step}-test/render/{submesh_type}/{batch['index'][0]}.png")

    def on_test_epoch_end(self):
        for submesh_type in ['target', 'other']:
            viewpath = f"it{self.true_global_step}-test/view/{submesh_type}"
            self.save_gif(viewpath, fps=30)