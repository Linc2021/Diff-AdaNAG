# Diff-AdaNAG
The official implementation of "The strength of Nesterov’s Accelerated Gradient in boosting transferability of stealthy adversarial attacks"


## Introduction 
Diff-AdaNAG, a novel framework that introduces Nesterov’s Accelerated Gradient (NAG) into diffusion-based adversarial example generation. Specifically, the diffusion mechanism guides the generation process toward the natural data distribution, achieving stealthy attacks with imperceptible adversarial examples. Meanwhile, an adaptive step-size strategy is utilized to harness the strong acceleration and
generalization capabilities of NAG in optimization, enhancing black-box transferability in adversarial attacks.

## Dependencies
```
python >= 3.11
torch, torchvision, torchaudio
numpy
scipy
torchattacks
matplotlib
tqdm
transformers
timm
```

## Preparation

1. Download the [DM checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and put it into `ckpt/`

2. Prepare imagenet dataset, and put it into `ILSVRC2012_img_val/`
- The ImageNet validation dataset can be download from [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/index.php)
- Use `valprep.sh` in `ILSVRC2012_img_val/` to preprocess the dataset.

## Example 
Compare with orther attack method:
```
python test_code.py -n 20 -d cuda:0 --save_adv_images
python additional_test_code.py -n 20 -d cuda:0 --save_adv_images
```
The all parameters for test_code.py:
```
[sdetime]:              sde time (default: 1)
[respace]:              ddim step (default: 'ddim50')
[nstep]:                attack step (default: 10)
[device]:               device (default: 'cuda:0')
[original_modelname]:   list of original models to use (default: None)
[target_modelname]:     list of target models to use (default: None)
[dataset]:              path for dataset (default: './ILSVRC2012_img_val')
[batchsize]:            batch size (default: 16)
[N_WORKERS]:            num_workers for DataLoader (default: 4)
[N_EXAMPLES]:           number of samples used in attack (default: 500)
[seed]:                 random seed (default: 42)
[save_adv_images]       save adversarial images (default: False)
[save_dir]              directory to save adversarial images (default: './adv_images')
```

Compare with i-FGSM with different mechanisms (DI, SI and TI)
```
python test_code_siditi.py -n 20 -d cuda:0 --SI --save_adv_images
```
The all parameters for test_code_siditi.py:
```
[sdetime]:             sde time (default: 1)
[respace]:             ddim step (default: 'ddim50')
[nstep]:               attack step (default: 10)
[device]:              device (default: 'cuda:0')
[original_modelname]:  list of original models to use (default: None)
[target_modelname]:    list of target models to use (default: None)
[SI]:                  enable SI
[DI]:                  enable DI
[TI]:                  enable TI
[dataset]:             path for dataset (default: './ILSVRC2012_img_val')
[batchsize]:           batch size (default: 16)
[N_WORKERS]:           num_workers for DataLoader (default: 4)
[N_EXAMPLES]:          number of samples used in attack (default: 500)
[seed]:                random seed (default: 42)
[save_adv_images]       save adversarial images (default: False)
[save_dir]              directory to save adversarial images (default: './adv_images')
```


## Reference

The code for adcversarial methods are inspired by [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch).

The code `load_dm.py` and code in `guided_diffusion/` are forked from [Diff-PGD](https://github.com/xavihart/Diff-PGD).



