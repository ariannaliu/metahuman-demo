export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="/home/zhixuan/demo/metahuman-demo/data/cctv"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_NAME \
  --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=50 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --rank=64 \
  --validation_epochs=5 \
  --output_dir="../checkpoints/sd-cctv-model2-lora-sdxl" \
  --validation_prompt="a CCTV-TVCC40011 camera in pink color" \
  --report_to="wandb"