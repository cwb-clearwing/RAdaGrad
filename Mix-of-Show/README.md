This is an experiment applying RGD like optimizers to EDLoRA pipeline.

Step-1: Clone Mix-of-Show repo.
```bash
git clone https://github.com/TencentARC/Mix-of-Show.git
```

Step-2: Clone  this repo and replace all files into Mix-of-show repo.
```bash
git clone https://github.com/cwb-clearwing/RGD_lora.git
```

Step-3: Train Mix-of-show single concept task using different optimziers(AdamW, scaled GD, scaled AdamW, RGD, RAdaGrad).
```bash
accelerate launch train_edlora.py -opt options/train/EDLoRA/real/8101_EDLoRA_potter_Cmix_B4_Repeat500.yml --optimzer radagrad
```

Step-4: Test Mix-of-show single concept LoRA.
```bash
python test_edlora.py -opt options/test/EDLoRA/human/8101_EDLoRA_potter_Cmix_B4_Repeat500.yml
```
