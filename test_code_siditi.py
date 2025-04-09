import random
import torch
import os
import csv
import numpy as np
from torchvision.models import (resnet50, resnet34, resnet18, resnet101, efficientnet_b0, googlenet,
    inception_v3, mnasnet0_5, mobilenet_v3_small, shufflenet_v2_x0_5, squeezenet1_1, vgg11, densenet121)

model = [resnet50, resnet34, resnet18, resnet101, efficientnet_b0, googlenet, inception_v3, mnasnet0_5, mobilenet_v3_small,
              shufflenet_v2_x0_5, squeezenet1_1, vgg11]
model_name = ["resnet50", "resnet34", "resnet18", "resnet101", "efficientnet_b0", "googlenet", "inception_v3", "mnasnet0_5", "mobilenet_v3_small",
              "shufflenet_v2_x0_5", "squeezenet1_1", "vgg11"]

from torchvision import transforms
from torchvision import datasets
import argparse
from adv_fun import AdaMSIFGM, diff_AdaNAG, diff_PGD, PGD, NIFGSM, FGSM
from utils import load_original_model, load_target_model, report_success_rate

parser = argparse.ArgumentParser(description='sim')
parser.add_argument('--sdetime','-s',type=int, default = 1, help="sde time")
parser.add_argument('--respace','-r', type=str , default='ddim50',help='ddim step')
parser.add_argument('--nstep','-n', type=int , default=10, help='attack step')
parser.add_argument('--device','-d', type=str , default='cuda:0', help='device')
parser.add_argument('--original_modelname', '-o', nargs='*', help='List of models to use')
parser.add_argument('--target_modelname', '-t', nargs='*', help='List of models to use')
parser.add_argument("--SI", action="store_true", help="Enable SI")
parser.add_argument("--DI", action="store_true", help="Enable DI")
parser.add_argument("--TI", action="store_true", help="Enable TI")
parser.add_argument('--dataset', type=str , default='./ILSVRC2012_img_val', help='Path for dataset')
parser.add_argument('--batchsize', type=int , default=16, help='Batchsize')
parser.add_argument('--N_WORKERS', type=int , default=4, help='num_workers for DataLoader')
parser.add_argument('--N_EXAMPLES', type=int , default=500, help='Number of samples used in attack')
parser.add_argument('--seed', type=int , default=42, help='random seed')
args = parser.parse_args()




DATA_PATH = args.dataset
BATCH_SIZE_TEST= args.batchsize
N_WORKERS = args.N_WORKERS
N_EXAMPLES = args.N_EXAMPLES


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)



    
# We can now compare with the vanilla attack
if __name__ == "__main__":
    device = args.device

    # Dataloader
    # Loaders should load unnormalized data (in [0,1]). Here, the test loader is a random subset of 500 test examples.
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    testset = datasets.ImageFolder(DATA_PATH, transform_test)
    indices = torch.from_numpy(np.random.choice(len(testset), size=(N_EXAMPLES,),
                                                replace=False))

    testsubset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(testsubset, batch_size=BATCH_SIZE_TEST,
                                             shuffle=False, num_workers=N_WORKERS,
                                             pin_memory=False)

    respace = args.respace
    t = args.sdetime
    nstep = args.nstep
    file = "all_model"
    original_models = model_name
    target_models = model_name
    fgsm_task_name = "FGSM"

    if args.original_modelname:
        original_models = args.original_modelname
        # run_model = [model[model_name.index(m)] for m in model_name]
        file = "source" + "_".join(original_models)
    if args.target_modelname:
        target_models = args.target_modelname
        # run_model = [model[model_name.index(m)] for m in model_name]
        file = file + "target" + "_".join(target_model)

    if args.SI:
        file = file + "_SI"
        fgsm_task_name = fgsm_task_name + "_SI"
    if args.TI:
        file = file + "_TI"
        fgsm_task_name = fgsm_task_name + "_TI"
    if args.DI:
        file = file + "_DI"
        fgsm_task_name = fgsm_task_name + "_DI"


    losses_dir = f'losses_{respace}_{t}_nstep{nstep}_{file}'
    os.makedirs(losses_dir, exist_ok=True)
    print(losses_dir)
    csv_file = os.path.join(losses_dir, f'losses_{respace}_{t}.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Attack', 'Original', 'Target', 'Step', 'Loss', 'Success Rate'])

    for original, original_name in zip(model, model_name):
        if original_name in original_models:
            for target, target_name in zip(model, model_name):
                if target_name in target_models:
                    base_model = load_original_model(original, device)
                    target_model = load_target_model(target, device)
                    print(original_name, target_name)

                    atk1 = FGSM(base_model, eps=4/255, steps = nstep, alpha = 4/255/nstep, SI = args.SI, TI = args.TI, diversity = args.DI)
                    success_rate1, losses1 = report_success_rate(atk1, target_model, testloader, device, nstep)
                    del atk1
                    torch.cuda.empty_cache()
                    
                    atk2 = diff_AdaNAG(base_model, eps=4/255, eta=0.05, steps=nstep, delta=1e-16, t=t, respace = respace,
                                            SI = args.SI, TI = args.TI, diversity = args.DI)
                    success_rate2, losses2 = report_success_rate(atk2, target_model, testloader, device, nstep)
                    del atk2
                    torch.cuda.empty_cache()


                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for idx, (loss, sr) in enumerate(zip(losses1, success_rate1)):
                            writer.writerow([fgsm_task_name, original_name, target_name, idx + 1, loss.item(), sr])
                        for idx, (loss, sr) in enumerate(zip(losses2, success_rate2)):
                            writer.writerow(['diff_AdaNAG', original_name, target_name, idx + 1, loss.item(), sr])



                    print("original={}, target={}".format(original_name, target_name))
                    print('{} Success rate: {:.2f}%\n'.format(fgsm_task_name, success_rate1[-1]))
                    print('Diff_AdaNAG Success rate: {:.2f}%\n'.format(success_rate2[-1]))


                    with open(f'log_{respace}_{t}_nstep{nstep}_{file}.txt', 'a') as f:
                        f.write("\noriginal={}, target={}\n".format(original_name, target_name))
                        f.write('{} Success rate: {:.2f}%\n'.format(fgsm_task_name, success_rate1[-1]))
                        f.write('Diff_AdaNAG Success rate: {:.2f}%\n'.format(success_rate2[-1]))
