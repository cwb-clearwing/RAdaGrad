This is a repo applying RGD like optimizers to diffusers pipeline.

1. Clone the diffusers repository:
```bash 
git clone https://github.com/huggingface/diffusers.git
```
2. Clone this repo and replace the files in peft folder and text_to_image folder to your local env. The peft folder is located in python site-packages while the text_to_image folder is located in the diffusers folder.
```bash 
git clone https://github.com/cwb-clearwing/RAdaGrad.git
cp -rf peft /your/local/peft/path/peft/
cp -rf text_to_iamge /your/diffusers/path/text_to_iamge
```
3. Train the model with accelerate command through VS code.
```bash
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "module": "accelerate.commands.launch",
            "args": [
                "D:\\diffusers\\examples\\text_to_image\\train_text_to_image_lora.py",
                "--pretrained_model_name_or_path", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "--dataset_name", "lambdalabs/naruto-blip-captions",
                "--resolution", "512",
                "--center_crop", 
                "--random_flip",
                "--checkpointing_steps", "100",
                "--train_batch_size", "1",
                "--gradient_accumulation_steps", "4",
                "--max_train_steps", "15000",
                "--learning_rate", "1e-04",
                "--max_grad_norm", "1",
                "--lr_scheduler", "cosine",
                "--lr_warmup_steps", "0",
                "--output_dir", ".\\naruto",
                "--hub_model_id", "naruto-lora",
                "--checkpointing_steps", "500",
                "--validation_prompt", "A naruto with blue eyes.",
                "--seed", "1337",
                "--mixed_precision", "no"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
          },     
    ]
```
Or though Bash
```bash
cd ${HOME}/diffusers/examples/text_to_image
#export conda_env=${HOME}/anaconda3/envs/test
export conda_env=${HOME}/anaconda3/envs/lora_prgd
export PATH=${conda_env}/bin:${HOME}/anaconda3/condabin:${PATH}
export LD_LIBRARY_PATH=${conda_env}/lib:${LD_LIBRARY_PATH}

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR=${HOME}/sddata/finetune/lora/naruto/
export HUB_MODEL_ID="naruto-lora"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
#export DATASET_NAME="pranked03/flowers-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --push_to_hub \
  --output_dir=${OUTPUT_DIR} \
  --mixed_precision=no \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="s1" \
  --seed=1337
```
4. Test the model. You can use these code to test your LoRA just like diffusers.
```python
import torch
from diffusers import AutoPipelineForText2Image
pipeline = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
for i in range(7):
  pipeline.load_lora_weights("/home/user/sddata/finetune/lora/naruto/checkpoint-"+str((i+2)*500), weight_name="pytorch_lora_weights.safetensors")
  #image = pipeline("a yellow flower").images[0]
  for j in range(20):
    image = pipeline("Hello Kitty with Naruto style, high quality, anime character, CG").images[0]
    image.save("./RGD_test/naruto_style_kitty-check-"+str((i+1)*500)+"-"+str(j)+".png")
  pipeline.unload_lora_weights()
```
