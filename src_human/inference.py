import cv2
import torch
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, StableDiffusionXLInpaintPipeline

from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

from diffusers.utils import load_image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline
)

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# image = cv2.imread("1.png")
image = cv2.imread("/home/zhixuan/demo/metahuman-demo/src_human/1.png")
faces = app.get(image)

faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

base_model_path = "SG161222/RealVisXL_V3.0"
# ip_ckpt = "ip-adapter-faceid_sdxl.bin"
ip_ckpt = "/home/zhixuan/demo/metahuman-demo/checkpoints/ipadapter/ip-adapter-faceid_sdxl.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
)

# load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

# generate image
prompt = "a man wearing a yellow working uniform"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
negative_prompt = ""

images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=1,
    width=1024, height=1024,
    num_inference_steps=30, guidance_scale=7.5
)

headshot = images[0]

resized_image = headshot.resize((512, 512), Image.LANCZOS)

# Create a new canvas of size (1024, 768) with a white background
canvas_size = (768, 1024)
canvas = Image.new('RGB', canvas_size, 'white')

# Calculate the position to paste the resized image (upper center)
x_offset = (canvas_size[0] - resized_image.width) // 2
y_offset = 0  # Upper center position

# Paste the resized image onto the canvas
canvas.paste(resized_image, (x_offset, y_offset))

# Create a mask for the outpainting region
# Initialize the mask with ones (areas to be outpainted)
mask_array = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)  # Note: Height x Width
# Set the area of the resized image to zeros (areas to keep)
mask_array[y_offset:y_offset + resized_image.height, x_offset:x_offset + resized_image.width] = 1
# Save the mask as an image (scale values to 0-255 for image representation)
mask_image = Image.fromarray(mask_array * 255)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
controlnet =ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnet, vae=vae
).to("cuda")

prompt = "the full body protrait of " + prompt
image = pipeline(
        prompt,
        height=1024,
        width=768,
        negative_prompt=negative_prompt,
        image=canvas,
        guidance_scale=4.5,
        num_inference_steps=25,
        controlnet_conditioning_scale=2.0,
        control_guidance_end=1.0,
    ).images[0]


################# Outpainting #################

pipeline_outpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
    "OzzyGT/RealVisXL_V4.0_inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=vae,
).to("cuda")

prompt = "high quality photo of " + prompt + ", highly detailed."

image = pipeline_outpaint(
        prompt,
        height=1024,
        width=768,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=10.0,
        strength=0.8,
        num_inference_steps=30,
    ).images[0]


image.save("final_output_demo.png")