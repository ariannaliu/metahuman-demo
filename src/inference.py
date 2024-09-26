from diffusers import DiffusionPipeline
import torch

lora_model_path = "../checkpoints/sd-cctv-model-lora-sdxl"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(lora_model_path)

prompt = "camera CCTV-TC70, white wall mount indoor camera in the sphere style."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("../outputs/result1.png")