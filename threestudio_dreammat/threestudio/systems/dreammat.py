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
        # BaseLift3DSystem의 optimizer 설정을 그대로 사용
        optimizers = super().configure_optimizers()
        
        # optimizers가 딕셔너리인 경우 첫 번째 optimizer를 가져옴
        optimizer = optimizers['optimizer']
        
        # 각 submesh 타입별로 별도의 optimizer 생성
        self.target_optimizer = optimizer  # 첫 번째 optimizer를 target에 사용
        self.other_optimizer = optimizer   # 동일한 optimizer를 other에도 사용
        return [self.target_optimizer, self.other_optimizer]

    def training_step(self, batch, batch_idx):
        # Get prompt processor outputs for both submesh types
        prompt_outputs = self.prompt_processor()

        # Process each submesh type independently
        for submesh_type in ["target", "other"]:
            # 해당 submesh의 optimizer 선택
            optimizer = self.target_optimizer if submesh_type == "target" else self.other_optimizer
            
            # 최적화 상태 초기화
            optimizer.zero_grad()
            
            # Get the corresponding batch data
            submesh_batch = batch[submesh_type]
            
            # Get prompt processor output for this submesh type
            prompt_utils = prompt_outputs[submesh_type]
            
            # Process the submesh
            out = self(submesh_batch)
            
            guidance_inp = out[submesh_type]["comp_rgb"]
            submesh_batch['cond_normal'] = out[submesh_type].get('comp_normal', None)
            submesh_batch['cond_depth'] = out[submesh_type].get('comp_depth', None)

            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **submesh_batch, rgb_as_latents=False, 
            )

            # Calculate loss for this submesh
            submesh_loss = 0.0    
            for name, value in guidance_out.items():
                self.log(f"train/{submesh_type}_{name}", value)
                if name.startswith("loss_"):
                    submesh_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

            for name, value in out.items():
                if name.startswith("loss_"):
                    self.log(f"train/{submesh_type}_{name}", value)
                    submesh_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{submesh_type}_{name}", self.C(value))

            # 각 submesh에 대해 독립적으로 backward 및 최적화 수행
            self.manual_backward(submesh_loss)
            optimizer.step()
            
            # Save training images if needed
            if self.cfg.save_train_image and self.true_global_step % self.cfg.save_train_image_iter == 0:
                train_images_row = [
                    {
                        "type": "rgb",
                        "img": out[submesh_type]["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
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
                        "img": out[submesh_type]["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out[submesh_type]["comp_depth"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
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
                ]
                train_conditions_row = [
                    {
                        "type": "grayscale",
                        "img": submesh_batch["condition_map"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 1:4],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 4:7],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 7:10],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 10:13],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 13:16],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 16:19],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": submesh_batch["condition_map"][0, :, :, 19:22],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                
                self.save_image_grid(
                    f"train/it{self.true_global_step}_{submesh_type}.png",
                    imgs=[train_images_row, train_conditions_row],
                    name=f"train_step_{submesh_type}",
                    step=self.true_global_step,
                )
            print(f"{submesh_type}_loss : ", submesh_loss)

            # 각 submesh의 loss를 독립적으로 반환
            if submesh_type == "target":
                target_loss = submesh_loss
            else:
                other_loss = submesh_loss

        return {
            "target_loss": target_loss,
            "other_loss": other_loss
        }

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