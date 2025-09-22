import random
import torch
import os
import csv
import gc
import numpy as np
from torchvision.models import vgg16, vgg19, inception_v3, resnet152


model_list = ["inception_v4", "inception_resnet_v2", vgg16, vgg19, inception_v3, resnet152 ]
model_name = ["inception_v4", "inception_resnet_v2", "vgg16", "vgg19", "inception_v3", "resnet152"]

attack_layers = {
    "inception_v4" : "1.features.5",
    "inception_v3" : "1.Mixed_5b",
    "resnet152" : "1.layer2",
    "vgg16" : "1.features.15",
    "vgg19" : "1.features.17",
    "inception_resnet_v2" : "1.conv2d_4a"
}
MFAA_attack_layers = {
    "inception_v4" : ["1.features.21", "1.features.18", "1.features.10", "1.features.5"],
    "inception_v3" : ["1.Mixed_7c", "1.Mixed_6e", "1.Mixed_5d", "1.Mixed_5b"],
    "resnet152" : ["1.layer4", "1.layer3.28", "1.layer3.18", "1.layer3.8", "1.layer2"],
    "vgg16" : ["1.features.29", "1.features.22", "1.features.15"],
    "vgg19" : ["1.features.35", "1.features.26", "1.features.17"],
    "inception_resnet_v2" : ["1.conv2d_7b", "1.mixed_7a", "1.mixed_6a", "1.conv2d_4a"]
}

from torchvision import transforms
from torchvision import datasets
import argparse
from adv_fun import diff_AdaNAG
from additional_adv import MFAA, NEAA, FIA
from utils import load_original_model, load_target_model, report_success_rate

parser = argparse.ArgumentParser(description='sim')
parser.add_argument('--sdetime','-s',type=int, default = 1, help="sde time")
parser.add_argument('--respace','-r', type=str , default='ddim50',help='ddim step')
parser.add_argument('--nstep','-n', type=int , default=10, help='attack step')
parser.add_argument('--device','-d', type=str , default='cuda:0', help='device')
parser.add_argument('--original_modelname', '-o', nargs='*', help='List of models to use')
parser.add_argument('--target_modelname', '-t', nargs='*', help='List of models to use')
parser.add_argument('--dataset', type=str , default="./ILSVRC2012_img_val", help='Path for dataset')
parser.add_argument('--batchsize', type=int , default=16, help='Batchsize')
parser.add_argument('--N_WORKERS', type=int , default=4, help='num_workers for DataLoader')
parser.add_argument('--N_EXAMPLES', type=int , default=500, help='Number of samples used in attack')
parser.add_argument('--seed', type=int , default=42, help='random seed')
parser.add_argument('--save_adv_images', action='store_true', help='Save adversarial images')
parser.add_argument('--save_dir', type=str, default='./additional_adv_images', help='Directory to save adversarial images')

args = parser.parse_args()




DATA_PATH = args.dataset
BATCH_SIZE_TEST= args.batchsize
N_WORKERS = args.N_WORKERS
N_EXAMPLES = args.N_EXAMPLES
device = args.device

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(args.seed)

    
# We can now compare with the vanilla attacks
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
    # select a subset of 500 examples. The original paper selects only original
    # examples that are correctly predicted by the target model, which might explain
    # why we obtain slightly different results.

    testsubset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(testsubset, batch_size=BATCH_SIZE_TEST,
                                             shuffle=False, num_workers=N_WORKERS,
                                             pin_memory=False)

    respace = args.respace
    t=args.sdetime
    nstep = args.nstep
    file = "all_model"
    original_models = model_name
    target_models = model_name

    if args.original_modelname:
        original_models = args.original_modelname
        # run_model = [model[model_name.index(m)] for m in model_name]
        file = "source" + "_".join(original_models)
    if args.target_modelname:
        target_models = args.target_modelname
        # run_model = [model[model_name.index(m)] for m in model_name]
        file = file + "_".join(target_models)

    losses_dir = f'losses_{respace}_{t}_nstep{nstep}_{file}'
    os.makedirs(losses_dir, exist_ok=True)
    print(losses_dir)
    csv_file = os.path.join(losses_dir, f'additional_losses_{respace}_{t}.csv')
    # 创建 CSV 文件并写入表头（如果不存在）
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Attack', 'Original', 'Target', 'Step', 'Loss', 'Success Rate'])

    for original, original_name in zip(model_list, model_name):
        if original_name in original_models:
            base_model = load_original_model(original, device)
            print(f"Generating adversarial examples with source model: {original_name}")

            target_models_dict = {}
            target_model_names = []
            for target, target_name in zip(model_list, model_name):
                if target_name in target_models:
                    # target_models_dict[target_name] = target
                    target_models_dict[target_name] = load_target_model(target, device)
                    target_model_names.append(target_name)

            save_dir_base = args.save_dir if args.save_adv_images else None
            
            atk1 = FIA(base_model, feature_layer=attack_layers[original_name], eps=4/255, alpha=4/255/nstep, steps=nstep)
            save_dir1 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_{nstep}_FIA") if save_dir_base else None
            success_rates_dict1, losses1 = report_success_rate(atk1, target_models_dict, testloader, device, nstep,
                                                        save_adv_images=args.save_adv_images, save_dir=save_dir1, N_EXAMPLES = N_EXAMPLES)
            del atk1
            torch.cuda.empty_cache()
                    
            atk3 = NEAA(base_model, feature_layer=attack_layers[original_name], eps=4/255, alpha=4/255/nstep, steps=nstep)
            save_dir3 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_{nstep}_NEAA") if save_dir_base else None
            success_rates_dict3, losses3 = report_success_rate(atk3, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir3, N_EXAMPLES = N_EXAMPLES)
            del atk3
            torch.cuda.empty_cache()

            atk4 = MFAA(base_model, eps=4/255, feature_layers=MFAA_attack_layers[original_name], alpha=4/255/nstep, steps=nstep)
            save_dir4 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_{nstep}_MFAA") if save_dir_base else None
            success_rates_dict4, losses4 = report_success_rate(atk4, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir4, N_EXAMPLES = N_EXAMPLES)
            del atk4
            torch.cuda.empty_cache()

            atk2 = diff_AdaNAG(base_model, eps=4/255, eta=0.05, steps=nstep, delta=1e-16, t=t, respace = respace)
            save_dir2 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_{nstep}_diff_AdaNAG") if save_dir_base else None
            success_rates_dict2, losses2 = report_success_rate(atk2, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir2, N_EXAMPLES = N_EXAMPLES)
            del atk2
            torch.cuda.empty_cache()

            atk5 = CAAM(base_model, eps=4/255, alpha=4/255/nstep, steps=nstep)
            save_dir5 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_{nstep}_CAAM") if save_dir_base else None
            success_rates_dict5, losses5 = report_success_rate(atk5, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir5, N_EXAMPLES = N_EXAMPLES)
            del atk5
            torch.cuda.empty_cache()

            for target_name in target_model_names:
                # Get results for this target model
                success_rate1 = success_rates_dict1[target_name]
                success_rate2 = success_rates_dict2[target_name]
                success_rate3 = success_rates_dict3[target_name]
                success_rate4 = success_rates_dict4[target_name]
                success_rate5 = success_rates_dict5[target_name]
                
                print(original_name, target_name)


                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for idx, (loss, sr) in enumerate(zip(losses1, success_rate1)):
                        writer.writerow(['FIA', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses2, success_rate2)):
                        writer.writerow(['diff_AdaNAG', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses3, success_rate3)):
                        writer.writerow(['NEAA', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses4, success_rate4)):
                        writer.writerow(['MFAA', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses5, success_rate5)):
                        writer.writerow(['CAAM', original_name, target_name, idx + 1, loss.item(), sr])


                    print("original={}, target={}".format(original_name, target_name))
                    print('FIA Success rate: {:.2f}%\n'.format(success_rate1[-1]))
                    print('diff_AdaNAG Success rate: {:.2f}%\n'.format(success_rate2[-1]))
                    print('NEAA Success rate: {:.2f}%\n'.format(success_rate3[-1]))
                    print('MFAA Success rate: {:.2f}%'.format(success_rate4[-1]))
                    print('CAAM Success rate: {:.2f}%'.format(success_rate5[-1]))

                    with open(f'additional_log_{respace}_{t}_nstep{nstep}_{file}.txt', 'a') as f:
                        f.write("\noriginal={}, target={}\n".format(original_name, target_name))
                        f.write('FIA Success rate: {:.2f}%\n'.format(success_rate1[-1]))
                        f.write('diff_AdaNAG Success rate: {:.2f}%\n'.format(success_rate2[-1]))
                        f.write('NEAA Success rate: {:.2f}%\n'.format(success_rate3[-1]))
                        f.write('MFAA Success rate: {:.2f}%\n'.format(success_rate4[-1]))
                        f.write('CAAM Success rate: {:.2f}%\n'.format(success_rate5[-1]))
            del base_model
            torch.cuda.empty_cache()
            gc.collect()  # 强制垃圾回收
            
