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
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Script to generate images using IPAdapter and ControlNet')
    parser.add_argument('--image_path', type=str, default='human_id/1.png', help='Path to the input image')
    parser.add_argument('--prompt', type=str, default='a man wearing yellow working uniform' ,help='Prompt for image generation')
    args = parser.parse_args()

    image_path = args.image_path
    prompt = args.prompt

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(image_path)
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    base_model_path = "SG161222/RealVisXL_V3.0"
    ip_ckpt = "/scratch/zhixuan2/creativeAI/ipadapter/ip-adapter-faceid_sdxl.bin"
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
    negative_prompt = "hands, arms, mask, monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    images = ip_model.generate(
        prompt= "a full face photo of "+prompt+",above neck, close up photo, with a clean background, good looking", negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=1,
        # prompt= "a close up photo of "+prompt+",above neck", negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=1,
        width=1024, height=1024,
        num_inference_steps=30, guidance_scale=7.5
    )

    headshot = images[0]
    os.makedirs("debug", exist_ok=True)
    headshot.save("debug/headshot.png")

    del ip_model
    del pipe
    torch.cuda.empty_cache()

    ################# Make Canvas #################

    # resized_image = headshot.resize((384, 384), Image.LANCZOS)
    # resized_image = headshot.resize((256, 256), Image.LANCZOS)
    resized_image = headshot.resize((189, 189), Image.LANCZOS)

    # Create a new canvas of size (1024, 768) with a white background
    canvas_size = (768, 1024)
    canvas = Image.new('RGB', canvas_size, 'white')

    # Calculate the position to paste the resized image (upper center)
    x_offset = (canvas_size[0] - resized_image.width) // 2
    y_offset = 64  # Upper center position

    # Paste the resized image onto the canvas
    canvas.paste(resized_image, (x_offset, y_offset))
    canvas.save("debug/canvas.png")

    # # Create a mask for the outpainting region
    # # Initialize the mask with ones (areas to be outpainted)
    # mask_array = np.ones((canvas_size[1], canvas_size[0]), dtype=np.uint8)  # Note: Height x Width
    # # Set the area of the resized image to zeros (areas to keep)
    # mask_array[y_offset:y_offset + resized_image.height, x_offset:x_offset + resized_image.width] = 0
    # # Save the mask as an image (scale values to 0-255 for image representation)
    # mask_image = Image.fromarray(mask_array * 255)

    ################# ControlNet #################

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")


    # controlnet = ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16")
    # pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnet, vae=vae
    # ).to("cuda")

    # prompt = "a full body image of "+ prompt + " with a clean background, good looking, high quality, high resolution"
    # image = pipeline(
    #         prompt,
    #         height=1024,
    #         width=768,
    #         negative_prompt=negative_prompt,
    #         image=canvas,
    #         guidance_scale=4.5,
    #         num_inference_steps=25,
    #         controlnet_conditioning_scale=1.5,
    #         control_guidance_end=1.0,
    #     ).images[0]

    # image.save("metahuman_output.png")



    controlnets = [
        ControlNetModel.from_pretrained(
            "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
        ),
        ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
        ),
    ]
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnets, vae=vae
    ).to("cuda")
    prompt = "a full body image of "+ prompt + " with a clean background, good looking, high quality, high resolution"

    # Resize the openpose image to match the canvas dimensions
    openpose_image = Image.open("human_id/tpose_01.png")
    openpose_image = openpose_image.resize((768, 1024), Image.LANCZOS)
    print(canvas.size)

    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=[canvas, openpose_image],
        # image=[openpose_image],
        guidance_scale=4.5,
        num_inference_steps=25,
        controlnet_conditioning_scale=[1.5, 1.5],
        control_guidance_start=[0.0,0.0],
        control_guidance_end=[1.0, 1.0]
    ).images[0]
    image.save("metahuman_output.png")


if __name__ == "__main__":
    main()