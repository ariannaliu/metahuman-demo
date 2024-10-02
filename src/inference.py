from diffusers import DiffusionPipeline
import torch
import argparse
import os

camera_dict = {}
camera_dict['PE204'] = 'cctv1'
camera_dict['Tapo C225'] = 'cctv2'
camera_dict['TSC-433P'] = 'cctv3'
camera_dict['CCTV-TC70'] = 'cctv4'
camera_dict['CCTV-RS PRO 146-4648'] = 'cctv5'
camera_dict['CCTV-TVCC40011'] = 'cctv6'
camera_dict['CCTV-TFT-22053D1H'] = 'cctv7'
camera_dict['CCTV-PNM-9000VD'] = 'cctv8'
camera_dict['CCTV-ASRIH686-025-22'] = 'cctv9'
camera_dict['CCTV-PNM-9320VQP'] = 'cctv10'
camera_dict['CCTV-PNM-9000VQ'] = 'cctv11'
camera_dict['CCTV-DS-2CE56C0T-IRMMF'] = 'cctv12'
camera_dict['CCTV-CD75-310-6527'] = 'cctv13'
camera_dict['ADT CCTV'] = 'cctv14'
camera_dict['PTZ q6125le'] = 'cctv15'
camera_dict['PTZ-B51N_800'] = 'cctv16'
camera_dict['PTZ-TC-A3555'] = 'cctv17'
camera_dict['PTZ-HD20A'] = 'cctv18'
camera_dict['Tapo C110'] = 'cctv19'



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to generate images using IPAdapter and ControlNet')
    parser.add_argument('--prompt', type=str, default='a photo of camera PE204 in pink color' ,help='Prompt for image generation')
    args = parser.parse_args()

    prompt = args.prompt
   # Check if any key from camera_dict exists in the prompt
    found_key = None
    for key in camera_dict.keys():
        if key in prompt:
            found_key = key
            break

    # Assert that a key has been found
    assert found_key is not None, "No matching camera model found in the prompt."

    lora_model_path = f"../checkpoints/camera/{camera_dict[found_key]}"
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(lora_model_path)
    image = pipe(prompt, num_inference_steps=50, guidance_scale=4.5).images[0]
    image.save("output_camera.png")