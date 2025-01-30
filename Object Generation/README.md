This is a repo applying RGD like optimizers to diffusers pipeline.

1. Clone the diffusers repository:
```bash 
git clone https://github.com/huggingface/diffusers.git
```
2. Clone this repo and replace the files in peft folder and text_to_image folder to your local env. The peft folder is located in python site-packages while the text_to_image folder is located in the diffusers folder.
```bash 
git clone https://github.com/cwb-clearwing/RGD_lora.git
cp -rf peft /your/local/peft/path/peft/
cp -rf text_to_iamge /your/diffusers/path/text_to_iamge
```
3. Train the model with accelerate command.
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
4. Test the model
