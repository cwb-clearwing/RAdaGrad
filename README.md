This is a repo for running RGD on diffusion model.

We applied the new algorithm to FC, LeNet-5, and U-Net, which required some modifications to the network structures. For FC, we replaced the linear layers with LowRankLinear, a linear layer that incorporates SVD and rank information, to apply the SVD decomposition from the new algorithm and perform compression based on rank. For LeNet-5, we replaced the convolutional layers (Conv2d) with Conv2d_Lr, a convolutional layer that also includes SVD and rank information, to achieve the same effect. Since U-Net is also composed of linear layers and convolutional layers, replacing its Linear layers and Conv2d layers allows it to utilize our new algorithm as well.

The RGD LoRA experiment used a modified operator file based on the framework provided by Huggingface. The experiment was divided into two control groups: the AdamW group and the RGD group, with three LoRA training targets. All experiments were conducted on an A100 GPU, with each training session taking approximately three hours. We selected open-source LoRA training datasets, namely naruto-blip-captions, flowers-blip-captions, and simpsons-blip-captions.

During the testing of these LoRAs, we chose "A hellokitty with naruto style" as the first test prompt, "A yellow flower" as the second test prompt, and "simpsons, a woman with long hair" as the third test prompt. The results show that our new RGD operator achieved the same training effectiveness as the original AdamW operator and even performed better on certain dimensions.

![RLora](https://github.com/user-attachments/assets/c827daef-33d5-44e9-8f3c-bd414561e88b)
