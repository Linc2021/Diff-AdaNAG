import random
import torch
import os
import csv
import gc
import numpy as np
from torchvision.models import resnet50, resnet34, resnet18, resnet101, efficientnet_b0, googlenet, inception_v3, mnasnet0_5, mobilenet_v3_small, shufflenet_v2_x0_5, squeezenet1_1, vgg11, densenet121


model_list = [resnet18, resnet34, resnet50, resnet101, efficientnet_b0, googlenet, inception_v3, mnasnet0_5, mobilenet_v3_small,
              shufflenet_v2_x0_5, squeezenet1_1, vgg11]
model_name = ["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_b0", "googlenet", "inception_v3", "mnasnet0_5", "mobilenet_v3_small",
              "shufflenet_v2_x0_5", "squeezenet1_1", "vgg11"]


from torchvision import transforms
from torchvision import datasets
import argparse
from adv_fun import AdaMSIFGM, diff_AdaNAG, diff_PGD, PGD, NIFGSM, SINIFGSM
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
parser.add_argument('--save_dir', type=str, default='./adv_images', help='Directory to save adversarial images')

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
    csv_file = os.path.join(losses_dir, f'losses_{respace}_{t}.csv')
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


            atk1 = PGD(base_model, eps=4/255, alpha=4/255/nstep, steps=nstep)
            save_dir1 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_PGD") if save_dir_base else None
            success_rates_dict1, losses1 = report_success_rate(atk1, target_models_dict, testloader, device, nstep,
                                                        save_adv_images=args.save_adv_images, save_dir=save_dir1, N_EXAMPLES = N_EXAMPLES)
            del atk1
            torch.cuda.empty_cache()

            atk2 = diff_AdaNAG(base_model, eps=4/255, eta=0.05, steps=nstep, delta=1e-16, t=t, respace = respace)
            save_dir2 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_diff_AdaNAG") if save_dir_base else None
            success_rates_dict2, losses2 = report_success_rate(atk2, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir2, N_EXAMPLES = N_EXAMPLES)
            del atk2
            torch.cuda.empty_cache()
                    
            atk3 = diff_PGD(base_model, eps=4/255, alpha=2/255, steps=nstep, t=t, respace = respace)
            save_dir3 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_diff_PGD") if save_dir_base else None
            success_rates_dict3, losses3 = report_success_rate(atk3, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir3, N_EXAMPLES = N_EXAMPLES)
            del atk3
            torch.cuda.empty_cache()

            atk4 = AdaMSIFGM(base_model, eps=4/255, alpha=1/255/nstep, steps=nstep, delta=1e-16)
            save_dir4 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_AdaMSIFGM") if save_dir_base else None
            success_rates_dict4, losses4 = report_success_rate(atk4, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir4, N_EXAMPLES = N_EXAMPLES)
            del atk4
            torch.cuda.empty_cache()

            atk5 = NIFGSM(base_model, eps=4/255, alpha=4/255/nstep, steps=nstep)
            save_dir5 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_NIFGSM") if save_dir_base else None
            success_rates_dict5, losses5 = report_success_rate(atk5, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir5, N_EXAMPLES = N_EXAMPLES)
            del atk5
            torch.cuda.empty_cache()

            atk6 = SINIFGSM(base_model, eps=4/255, alpha=4/255/nstep, steps=nstep)
            save_dir6 = os.path.join(save_dir_base, f"{nstep}", f"{original_name}_SINIFGSM") if save_dir_base else None
            success_rates_dict6, losses6 = report_success_rate(atk6, target_models_dict, testloader, device, nstep,
                                                                          save_adv_images=args.save_adv_images, save_dir=save_dir6, N_EXAMPLES = N_EXAMPLES)
            del atk6
            torch.cuda.empty_cache()


            for target_name in target_model_names:
                # Get results for this target model
                success_rate1 = success_rates_dict1[target_name]
                success_rate2 = success_rates_dict2[target_name]
                success_rate3 = success_rates_dict3[target_name]
                success_rate4 = success_rates_dict4[target_name]
                success_rate5 = success_rates_dict5[target_name]
                success_rate6 = success_rates_dict6[target_name]
                
                print(original_name, target_name)


                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for idx, (loss, sr) in enumerate(zip(losses1, success_rate1)):
                        writer.writerow(['PGD', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses2, success_rate2)):
                        writer.writerow(['diff_AdaNAG', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses3, success_rate3)):
                        writer.writerow(['diff_PGD', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses4, success_rate4)):
                        writer.writerow(['AdaMSI-FGM', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses5, success_rate5)):
                        writer.writerow(['NI-FGSM', original_name, target_name, idx + 1, loss.item(), sr])
                    for idx, (loss, sr) in enumerate(zip(losses6, success_rate6)):
                        writer.writerow(['SI-NI-FGSM', original_name, target_name, idx + 1, loss.item(), sr])



                    print("original={}, target={}".format(original_name, target_name))
                    print('PGD Success rate: {:.2f}%\n'.format(success_rate1[-1]))
                    print('diff_AdaNAG Success rate: {:.2f}%\n'.format(success_rate2[-1]))
                    print('diff_PGD Success rate: {:.2f}%\n'.format(success_rate3[-1]))
                    print('AdaMSI-FGM Success rate: {:.2f}%'.format(success_rate4[-1]))
                    print('NI-FGSM Success rate: {:.2f}%'.format(success_rate5[-1]))
                    print('SI-NI-FGSM Success rate: {:.2f}%'.format(success_rate6[-1]))

                    with open(f'log_{respace}_{t}_nstep{nstep}_{file}.txt', 'a') as f:
                        f.write("\noriginal={}, target={}\n".format(original_name, target_name))
                        f.write('PGD Success rate: {:.2f}%\n'.format(success_rate1[-1]))
                        f.write('diff_AdaNAG Success rate: {:.2f}%\n'.format(success_rate2[-1]))
                        f.write('diff_PGD Success rate: {:.2f}%\n'.format(success_rate3[-1]))
                        f.write('AdaMSI-FGM Success rate: {:.2f}%\n'.format(success_rate4[-1]))
                        f.write('NI-FGSM Success rate: {:.2f}%\n'.format(success_rate5[-1]))
                        f.write('SI-NI-FGSM Success rate: {:.2f}%\n'.format(success_rate6[-1]))
            del base_model
            torch.cuda.empty_cache()
            gc.collect()  # 强制垃圾回收
            
