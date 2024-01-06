import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

class Upscaler:
    def __init__(self, text2img: StableDiffusionPipeline, scale=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale = scale
        self.model = RealESRGAN(self.device, scale=self.scale)
        self.model.load_weights(f'weights/RealESRGAN_x{self.scale}.pth', download=True)
        self.img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        
    def upscale(self, imgs: list[Image]) -> list[Image]:
        torch.cuda.empty_cache()
        upscaled_imgs = []
        for img in imgs:
            upscaled_img = self.model.predict(img)
            upscaled_imgs.append(upscaled_img)
        return upscaled_imgs

    def hires_fix(self, imgs: list[Image], prompt: str, negative_prompt: str) -> list[Image]:
        upscaled_images = self.upscale(imgs)
        results = self.img2img(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            image=upscaled_images, 
            strength=0.2, 
            guidance_scale=7.5, 
            num_inference_steps=30, 
            schduler="EulerAncestralDiscreteScheduler",
        ).images
        return results
        