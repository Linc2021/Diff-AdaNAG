import numpy as np
import matplotlib.pyplot as plt
import json
import os
import timm
from typing import List, Union, Dict, Callable

import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm



# load model
# load the surrogate and target models.
def add_normalization_layer(model, mean, std):
    """
    Add a data normalization layer to a model
    """
    return torch.nn.Sequential(
        transforms.Normalize(mean=mean, std=std),
        model
    )
# Load the original model, the pretrained resnet18 model provided by torchvision.
def load_original_model(model: Union[str, Callable[[], nn.Module]], device):
    if callable(model):
        base_model = model(pretrained=True)
    elif isinstance(model, str):
        base_model = timm.create_model(model, pretrained=True)
    base_model = add_normalization_layer(model=base_model,
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    base_model = base_model.eval()
    # base_model = nn.DataParallel(base_model)
    base_model = base_model.to(device)
    return base_model

# load the target model, the pretrained resnet50 model provided by torchvision.
def load_target_model(model: Union[str, Callable[[], nn.Module]], device):
    if callable(model):
        target_model = model(pretrained=True)
    elif isinstance(model, str):
        target_model = timm.create_model(model, pretrained=True)
    target_model = add_normalization_layer(model=target_model,
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    target_model = target_model.eval()
    # target_model = nn.DataParallel(target_model)
    target_model = target_model.to(device)
    return target_model


# def report_success_rate(atk, target_model, testloader, device, nstep=1):
#     """
#     Compute the success rate of the provided attack on test images
#     """
#     correct = np.zeros(nstep)
#     total = np.zeros(nstep)
#     for images, labels in tqdm(testloader, total=len(testloader)):
#         images = images.to(device)
#         labels = labels.to(device)
#         adv_images, losses = atk(images, labels)

#         for i in range(len(adv_images)):
#             with torch.no_grad():
#                 outputs_adv = target_model(adv_images[i])
#                 _, predicted_adv = torch.max(outputs_adv.data, 1)
#             total[i] += labels.size(0)
#             correct[i] += (predicted_adv == labels).sum().item()
#     success_rates = 100 - 100 * (correct / total)
#     return success_rates.tolist(), losses

def report_success_rate(atk, target_models, testloader, device, nstep=1, save_adv_images=False, save_dir=None, N_EXAMPLES = 500):
    """
    Compute the success rate of the provided attack on test images
    If target_models is a dictionary, test against multiple target models and return results for all
    If target_models is a single model, test against that model and return results
    If save_adv_images is True, save the generated adversarial images to save_dir
    """
    if isinstance(target_models, dict):
        # Test against multiple target models
        results = {}
        
        # Generate adversarial examples once
        img_idx = 0
        for images, labels in tqdm(testloader, total=len(testloader)):
            images = images.to(device)
            labels = labels.to(device)
            adv_images, losses = atk(images, labels)
            
            # Save adversarial images if requested (only final step images, each batch item separately)
            if save_adv_images and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                # Get the final step adversarial images (last element in the list)
                if isinstance(adv_images, list) and len(adv_images) > 0:
                    final_adv_images = adv_images[-1]  # Tensor of shape [batch_size, C, H, W]
                    if torch.is_tensor(final_adv_images):
                        # Save each image in the batch separately
                        for batch_idx in range(final_adv_images.shape[0]):
                            save_path = os.path.join(save_dir, f"{str(img_idx + batch_idx).rjust(3, '0')}_adv_img.png")
                            torchvision.utils.save_image(final_adv_images[batch_idx], save_path)
                            save_path = os.path.join(save_dir, f"{str(img_idx + batch_idx).rjust(3, '0')}_clean_img.png")
                            torchvision.utils.save_image(images[batch_idx], save_path)
                img_idx += images.shape[0]  # Increment by batch size
            
            # Test each target model with the same adversarial examples
            # for model_name, model in target_models.items():
            for model_name, target_model in target_models.items():
                if model_name not in results:
                    results[model_name] = {'correct': np.zeros(nstep), 'total': np.zeros(nstep)}
                    # target_model = load_target_model(model, device)
                
                for i in range(len(adv_images)):
                    with torch.no_grad():
                        outputs_adv = target_model(adv_images[i])
                        _, predicted_adv = torch.max(outputs_adv.data, 1)
                    results[model_name]['total'][i] += labels.size(0)
                    results[model_name]['correct'][i] += (predicted_adv == labels).sum().item()
        
        
        # Calculate success rates for all target models
        success_rates_dict = {}
        for model_name, result in results.items():
            success_rates = 100 - 100 * (result['correct'] / result['total'])
            success_rates_dict[model_name] = success_rates.tolist()
            
        return success_rates_dict, losses

    
def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    imagnet_data = image_folder_custom_label(root='./data/imagenet', 
                                             transform=transform,
                                             idx2label=idx2label)
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
    print("Used normalization: mean=", MEAN, "std=", STD)
    return iter(data_loader).next()

def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def image_folder_custom_label(root, transform, idx2label) :
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2


@torch.no_grad()
def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return (100 * float(correct) / total)
