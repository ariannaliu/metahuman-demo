import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
image_path = "/scratch/zhixuan2/creativeAI/metahuman/src_human/metahuman_outputs/image_frontview.png"
# image_path = "/scratch/zhixuan2/creativeAI/metahuman/src_human/metahuman_outputs/image_backview.png"
low_res_img = Image.open(image_path).convert("RGB")
prompt = 'a man wearing yellow working uniform'
prompt = "a full body image of "+ prompt + " with a clean background, good looking, high quality, high resolution"
# prompt = "a back view image of "+ prompt 
low_res_img = low_res_img.resize((int(low_res_img.size[0]/2), int(low_res_img.size[1]/2)))

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("metahuman_outputs/higher_fronview.png")