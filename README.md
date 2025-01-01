This is a repo for running RGD on diffusion model.

The RGD LoRA experiment used a modified operator file based on the framework provided by Huggingface. The experiment was divided into two control groups: the AdamW group and the RGD group, with three LoRA training targets. All experiments were conducted on an A100 GPU, with each training session taking approximately three hours. We selected open-source LoRA training datasets, namely naruto-blip-captions, flowers-blip-captions, and simpsons-blip-captions.

During the testing of these LoRAs, we chose "A hellokitty with naruto style" as the first test prompt, "A yellow flower" as the second test prompt, and "simpsons, a woman with long hair" as the third test prompt. The results show that our new RGD operator achieved the same training effectiveness as the original AdamW operator and even performed better on certain dimensions.
