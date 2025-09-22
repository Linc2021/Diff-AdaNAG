# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from typing import List, Union, Dict

import torch
from torchattacks.attack import Attack
import torch.nn as nn
import torch.nn.functional as F


def gkern(kernlen: int = 15, nsig: float = 3.0) -> np.ndarray:
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = torch.from_numpy(np.exp(-0.5 * (x / 1.0) ** 2) / math.sqrt(2 * math.pi)).float()
    kern1d = (kern1d / kern1d.sum()).numpy()
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)

def project_kern(kern_size: int = 3) -> np.ndarray:
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    return kern

def depthwise_kernel_from_2d(k2d: np.ndarray, channels: int) -> torch.Tensor:
    k = torch.from_numpy(k2d).float()
    return k.view(1, 1, *k.shape).repeat(channels, 1, 1, 1)  # (C,1,H,W)


def stack_to_depthwise(kernel_2d: np.ndarray, channels: int) -> torch.Tensor:
    k = torch.from_numpy(kernel_2d).float()
    k = k.view(1, 1, *k.shape).repeat(channels, 1, 1, 1)  # (C,1,H,W)
    return k

def input_diversity(x: torch.Tensor, resize_rate: float, prob: float) -> torch.Tensor:
    if prob <= 0:
        return x
    N, C, H, W = x.shape
    img_resize = int(max(H, W) * resize_rate)
    if resize_rate < 1:
        low, high = img_resize, max(H, W)
    else:
        low, high = max(H, W), img_resize
    if low == high:
        return x
    rnd = torch.randint(low=low, high=high + 1, size=(1,), device=x.device).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = torch.randint(low=0, high=h_rem + 1, size=(1,), device=x.device).item()
    pad_left = torch.randint(low=0, high=w_rem + 1, size=(1,), device=x.device).item()
    pad_bottom = h_rem - pad_top
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0.0)
    padded = F.interpolate(padded, size=[H, W], mode='bilinear', align_corners=False)
    return padded if torch.rand(1, device=x.device).item() < prob else x

def l2_normalize_per_sample(t: torch.Tensor, dims=(1,2,3), eps: float = 1e-12) -> torch.Tensor:
    norm = torch.sqrt((t ** 2).sum(dim=dims, keepdim=True) + eps)
    return t / norm

def mean_abs_normalize_per_sample(t: torch.Tensor, dims=(1,2,3), eps: float = 1e-12) -> torch.Tensor:
    scale = t.abs().mean(dim=dims, keepdim=True) + eps
    return t / scale

def resolve_module(model: nn.Module, name: str) -> nn.Module:
    cur = model
    for attr in name.split('.'):
        if attr.isdigit():
            cur = cur[int(attr)]
        else:
            cur = getattr(cur, attr)
    return cur

class FeatureGrabber:
    def __init__(self, model: nn.Module, layers: List[Union[str, nn.Module]]):
        self.model = model
        self.layers = layers
        self.handles = []
        self.feats: Dict[str, torch.Tensor] = {}

        for l in layers:
            if isinstance(l, str):
                m = resolve_module(model, l)
                key = l
            else:
                m = l
                key = str(id(l))
            handle = m.register_forward_hook(self._hook_fn(key))
            self.handles.append(handle)

    def _hook_fn(self, key):
        def fn(module, inp, out):
            # 统一为 (B,C,H,W)
            self.feats[key] = out
        return fn

    def clear(self):
        self.feats.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get(self, name_or_module) -> torch.Tensor:
        key = name_or_module if isinstance(name_or_module, str) else str(id(name_or_module))
        return self.feats[key]



class FIA(Attack):

    def __init__(self,
                 model: nn.Module,
                 feature_layer: Union[str, nn.Module],
                 eps: float = 16/255, alpha: float = 1.6/255, steps: int = 10, decay: float = 1.0,
                 ens: int = 30, keep_prob: float = 0.9,
                 # DI
                 diversity: bool = False, resize_rate: float = 250/224, diversity_prob: float = 0.7,
                 # TI
                 TI: bool = False, len_kernel: int = 15, nsig: float = 3.0,
                 # PIM
                 PIM: bool = False, amp_factor: float = 2.5, gamma_pim: float = 0.5, pkern: int = 3):
        super().__init__("FIA", model)
        self.supported_mode = ["default"]  

        self.feature_layer = feature_layer
        self.eps, self.alpha, self.steps, self.decay = eps, alpha, steps, decay
        self.ens, self.keep_prob = ens, keep_prob

        self.diversity, self.resize_rate, self.diversity_prob = diversity, resize_rate, diversity_prob
        self.TI = TI
        self.ti_kernel_2d = gkern(len_kernel, nsig)              
        self.PIM = PIM
        self.amp_factor, self.gamma_pim, self.pkern = amp_factor, gamma_pim, pkern
        self.pim_kernel_2d = project_kern(pkern)                

    def _compute_fia_weights(self, images: torch.Tensor, labels: torch.Tensor,
                            grabber: FeatureGrabber) -> torch.Tensor:
        self.model.eval()

        num_classes = self.get_logits(images[:1]).size(1)
        one_hot = F.one_hot(labels, num_classes=num_classes).float().to(self.device)

        acc = None

        if int(self.ens) == 0:
            x0 = images.clone().detach().to(self.device)
            x_in = input_diversity(x0, self.resize_rate, self.diversity_prob) if self.diversity else x0
            grabber.clear()
            x_in.requires_grad_(True)
            logits = self.get_logits(x_in)
            feat = grabber.get(self.feature_layer)
            loss_w = (logits * one_hot).sum()
            g = torch.autograd.grad(loss_w, feat, retain_graph=False, create_graph=False)[0].detach()
            acc = g.clone()

        for _ in range(int(self.ens)):
            mask = torch.bernoulli(torch.full_like(images, self.keep_prob, device=self.device))
            xm = images * mask
            x_in = input_diversity(xm, self.resize_rate, self.diversity_prob) if self.diversity else xm
            grabber.clear()
            x_in.requires_grad_(True)
            logits = self.get_logits(x_in)
            feat = grabber.get(self.feature_layer)
            loss_w = (logits * one_hot).sum()
            g = torch.autograd.grad(loss_w, feat, retain_graph=False, create_graph=False)[0].detach()
            acc = g if acc is None else (acc + g)

        weights = -l2_normalize_per_sample(acc)
        return weights

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        

        grabber = FeatureGrabber(self.model, [self.feature_layer])


        with torch.enable_grad():
            weights = self._compute_fia_weights(images, labels, grabber)   # (N,C,H,W)

        grabber.clear()
        
        momentum = torch.zeros_like(images, device=self.device)
        amp = torch.zeros_like(images, device=self.device)     

        ti_kernel = stack_to_depthwise(self.ti_kernel_2d, images.shape[1]).to(self.device)    # (C,1,k,k)
        pim_kernel = stack_to_depthwise(self.pim_kernel_2d, images.shape[1]).to(self.device)  # (C,1,k,k)
        pad_pim = self.pkern // 2
        pad_ti = ti_kernel.shape[-1] // 2

        adv = images.clone().detach()
        losses, all_images = [], []

        for _ in range(self.steps):
            adv.requires_grad_(True)
            x_in = input_diversity(adv, self.resize_rate, self.diversity_prob) if self.diversity else adv
            _ = self.get_logits(x_in)                         
            feat = grabber.get(self.feature_layer)            # feature(adv)

            # get_fia_loss: mean( feat * weights )  
            loss = (feat * weights).mean()
            losses.append(loss.detach())

            # d(loss)/d(adv)
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

            if self.TI:
                grad = F.conv2d(grad, ti_kernel, stride=1, padding=pad_ti, groups=grad.shape[1])

            grad_n = mean_abs_normalize_per_sample(grad)
            momentum = self.decay * momentum + grad_n

            if self.PIM:
                alpha_beta = self.alpha * self.amp_factor
                gamma = self.gamma_pim * alpha_beta

                amp = amp + alpha_beta * momentum.sign()
                cut = (amp.abs() - self.eps).clamp(min=0.0) * amp.sign()
                cut_pad = F.pad(cut, [pad_pim, pad_pim, pad_pim, pad_pim], value=0.0)
                proj = gamma * F.conv2d(cut_pad, pim_kernel, stride=1, padding=0, groups=cut.shape[1]).sign()
                amp = amp + proj

                adv = adv.detach() + alpha_beta * momentum.sign() + proj
            else:
                adv = adv.detach() + self.alpha * momentum.sign()

            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, min=0.0, max=1.0).detach()
            all_images.append(adv)

        grabber.close()
        return all_images, losses




class MFAA(Attack):
    def __init__(
        self, model: nn.Module,
        feature_layers: List[Union[str, nn.Module]],
        # feature_layers: List[Union[str, nn.Module]],
        eps: float = 16/255, alpha: float = 1.6/255, steps: int = 10, decay: float = 1.0,
        ens: int = 30, keep_prob: float = 0.8,
        diversity: bool = False, resize_rate: float = 250/224, diversity_prob: float = 0.7,
        TI: bool = False, len_kernel: int = 15, nsig: float = 3.0,
        PIM: bool = False, amp_factor: float = 2.5, gamma_pim: float = 0.5, pkern: int = 3
    ):
        super().__init__("MFAA", model)
        self.feature_layers = feature_layers
        self.eps, self.alpha, self.steps, self.decay = eps, alpha, steps, decay
        self.ens, self.keep_prob = ens, keep_prob

        self.diversity, self.resize_rate, self.diversity_prob = diversity, resize_rate, diversity_prob
        self.TI = TI
        self.stacked_kernel = stack_to_depthwise(gkern(len_kernel, nsig), channels=3) 
        self.PIM = PIM
        self.amp_factor, self.gamma_pim, self.pkern = amp_factor, gamma_pim, pkern
        self.pim_kernel = stack_to_depthwise(project_kern(pkern), channels=3)

    def _compute_attention_weights(self, images: torch.Tensor, labels: torch.Tensor,
                                   grabber: FeatureGrabber) -> Dict[str, torch.Tensor]:
        model = self.model
        model.eval()
        num_classes = self.get_logits(images).size(1)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()

        acc: Dict[str, torch.Tensor] = {}
        for l in self.feature_layers:
            key = l if isinstance(l, str) else str(id(l))
            acc[key] = None

        for _ in range(max(self.ens, 1)):
            if self.ens > 0:
                mask = torch.bernoulli(torch.full_like(images, self.keep_prob, device=images.device))
                x_m = images * mask
            else:
                x_m = images

            x_in = input_diversity(x_m, self.resize_rate, self.diversity_prob) if self.diversity else x_m
            grabber.clear()
            x_in.requires_grad_(True)
            logits = self.get_logits(x_in)
            loss_w = (logits * one_hot).sum()
            feats = [grabber.get(l) for l in self.feature_layers]  
            grads = torch.autograd.grad(loss_w, feats, retain_graph=False, create_graph=False, allow_unused=False)
            
            # acc = g if acc is None else (acc + g)
            for l, g in zip(self.feature_layers, grads):
                key = l if isinstance(l, str) else str(id(l))
                g = g.detach()
                acc[key] = g.clone() if acc[key] is None else (acc[key] + g)
            

        for l in self.feature_layers:
            key = l if isinstance(l, str) else str(id(l))
            acc[key] = -l2_normalize_per_sample(acc[key]) 

        return acc

    def _fia_loss(self, feat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (feat * weight).mean()

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv = images.clone().detach()
        momentum = torch.zeros_like(images, device=self.device)
        amp = torch.zeros_like(images, device=self.device)  # PIM amplification
        losses = []
        all_images = []

        grabber = FeatureGrabber(self.model, self.feature_layers)

        with torch.enable_grad():
            weights = self._compute_attention_weights(images, labels, grabber)

        for _ in range(self.steps):
            adv.requires_grad_(True)
            grabber.clear()

            x_in = input_diversity(adv, self.resize_rate, self.diversity_prob) if self.diversity else adv
            logits = self.get_logits(x_in)

            feats = [grabber.get(l) for l in self.feature_layers]
            ws = [weights[l if isinstance(l, str) else str(id(l))] for l in self.feature_layers]
            q = 1.0  
            cascade_losses = []

            L = self._fia_loss(feats[0], ws[0])
            cascade_losses.append(L)

            for i in range(1, len(feats)):
                g = torch.autograd.grad(cascade_losses[-1], feats[i],
                                        retain_graph=True, create_graph=False)[0].detach()
                w_corr = q * l2_normalize_per_sample(g) + ws[i]
                L = self._fia_loss(feats[i], w_corr)
                cascade_losses.append(L)
            
            loss_final = cascade_losses[-1]


            losses.append(loss_final.detach())

            grad = torch.autograd.grad(loss_final, adv, retain_graph=False, create_graph=False)[0]

            if self.TI:
                k = self.stacked_kernel.to(grad.device, dtype=grad.dtype)
                grad = F.conv2d(grad, k, stride=1, padding='same', groups=grad.shape[1])

            grad_n = mean_abs_normalize_per_sample(grad)
            momentum = self.decay * momentum + grad_n

            if self.PIM:
                alpha_beta = self.alpha * self.amp_factor
                amp = amp + alpha_beta * momentum.sign()
                cut_noise = (amp.abs() - self.eps).clamp(min=0.0) * amp.sign()
                pker = self.pim_kernel.to(grad.device, dtype=grad.dtype)
                projection = self.gamma_pim * F.conv2d(cut_noise, pker, stride=1, padding='same', groups=grad.shape[1])
                amp = amp + projection
                adv = adv.detach() + alpha_beta * momentum.sign() + projection
            else:
                adv = adv.detach() + self.alpha * momentum.sign()

            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv)

        grabber.close()
        return all_images, losses



class CAAM(Attack):
    def __init__(
        self, model: nn.Module,
        eps: float = 16/255, steps: int = 10, alpha: float = None, decay: float = 1.0,
        diversity: bool = False, resize_rate: float = 331/299, diversity_prob: float = 0.5,
        TI: bool = False, len_kernel: int = 7, nsig: float = 3.0,
        num_channel_aug: int = 3
    ):
        super().__init__("CAAM", model)
        self.eps, self.steps, self.decay = eps, steps, decay
        self.alpha = alpha if alpha is not None else eps / steps
        self.diversity, self.resize_rate, self.diversity_prob = diversity, resize_rate, diversity_prob
        self.TI = TI
        self.stacked_kernel = stack_to_depthwise(gkern(len_kernel, nsig), channels=3)
        self.num_channel_aug = num_channel_aug
        
    @torch.no_grad()
    def _channel_transformations(self, x: torch.Tensor, k: int = 3) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        outs = []
        for _ in range(k):
            padding = torch.empty(1, 1, H, W, device=x.device, dtype=x.dtype).uniform_(0, 1.0)
            perm = torch.randperm(H, device=x.device)
            padding = padding[:, :, perm, :]                 
            padding_b = padding.expand(B, 1, H, W)           
    
            
            R = x[:, 0:1]
            G = x[:, 1:2]
            Bc = x[:, 2:3]
    
            r_idx = torch.randint(0, 4, (B,), device=x.device)
            g_idx = torch.randint(0, 4, (B,), device=x.device)
            b_idx = torch.randint(0, 4, (B,), device=x.device)
    
            bag = torch.stack([padding_b, R, G, Bc], dim=1)  # (B, 4, 1, H, W)
            idx = torch.arange(B, device=x.device)
            newR = bag[idx, r_idx, :, :, :]                  # (B, 1, H, W)
            newG = bag[idx, g_idx, :, :, :]
            newB = bag[idx, b_idx, :, :, :]
    
            newx = torch.cat([newR, newG, newB], dim=1)      # (B, 3, H, W)
            outs.append(newx)
        return outs

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv = images.clone().detach()
        momentum = torch.zeros_like(images, device=self.device)
        losses = []
        all_images = []

        data_list = self._channel_transformations(images, self.num_channel_aug)
        data_list.append(images.clone())

        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            noise_all = torch.zeros_like(images, device=self.device)
            for v in data_list:
                scales = [1.0, 0.5, 0.25, 0.125, 0.0625]         
                weights = [1.0, 0.5, 0.25, 0.125, 0.0625]        
                noise_v = torch.zeros_like(images, device=self.device)

                for s, w in zip(scales, weights):
                    v_scaled = (v * s).detach().requires_grad_(True)   
                    inp = input_diversity(v_scaled, self.resize_rate, self.diversity_prob) if self.diversity else v_scaled
                    logits = self.get_logits(inp)
                    cost = -loss_fn(logits, target_labels) if self.targeted else loss_fn(logits, labels)
                    g = torch.autograd.grad(cost, v_scaled, retain_graph=False, create_graph=False)[0]
                    noise_v = noise_v + w * g      
                noise_v = noise_v / (noise_v.abs().mean(dim=(1,2,3), keepdim=True) + 1e-12)
                noise_all = noise_all + noise_v


            noise_all = mean_abs_normalize_per_sample(noise_all)
            if self.TI:
                k = self.stacked_kernel.to(noise_all.device, dtype=noise_all.dtype)
                noise_all = F.conv2d(noise_all, k, stride=1, padding='same', groups=noise_all.shape[1])
            momentum = self.decay * momentum + noise_all

            adv = adv.detach() + self.alpha * momentum.sign()
            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv)

            new_list = []
            for v in data_list:
                vv = torch.clamp(v + self.alpha * momentum.sign(), min=images - self.eps, max=images + self.eps)
                vv = vv.clamp(0, 1).detach()
                new_list.append(vv)
            data_list = new_list

            logits = self.get_logits(adv)
            cost = -loss_fn(logits, target_labels) if self.targeted else loss_fn(logits, labels)
            losses.append(cost.detach())

        return all_images, losses



class NEAA(Attack):
    def __init__(
        self, model: nn.Module,
        feature_layer: Union[str, nn.Module],
        eps: float = 16/255, alpha: float = 1.6/255, steps: int = 10, decay: float = 1.0,
        N_bases: int = 10, ens: int = 30, base_noise_std: float = 1.0, mix_noise_std: float = 0.2,
        gamma_neg: float = 1.0,
        diversity: bool = False, resize_rate: float = 331/299, diversity_prob: float = 0.7,
        TI: bool = False, len_kernel: int = 15, nsig: float = 3.0,
        PIM: bool = False, amp_factor: float = 2.5, gamma_pim: float = 0.5, pkern: int = 3
    ):
        super().__init__("NEAA", model)
        self.feature_layer = feature_layer
        self.eps, self.alpha, self.steps, self.decay = eps, alpha, steps, decay
        self.N_bases, self.ens = N_bases, ens
        self.base_noise_std, self.mix_noise_std = base_noise_std, mix_noise_std
        self.gamma_neg = gamma_neg

        self.diversity, self.resize_rate, self.diversity_prob = diversity, resize_rate, diversity_prob
        self.TI = TI
        self.stacked_kernel = stack_to_depthwise(gkern(len_kernel, nsig), channels=3)
        self.PIM = PIM
        self.amp_factor, self.gamma_pim, self.pkern = amp_factor, gamma_pim, pkern
        self.pim_kernel = stack_to_depthwise(project_kern(pkern), channels=3)

    def _prepare_weights_and_bases(self, images, labels, grabber: FeatureGrabber):
        model = self.model
        model.eval()
        B = images.size(0)
        num_classes = self.get_logits(images).size(1)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        key = self.feature_layer if isinstance(self.feature_layer, str) else str(id(self.feature_layer))

        weights_list = []
        basefeat_list = []

        for n in range(self.N_bases):
            # baseline：x_base = x + N(0, base_noise_std^2)
            x_base = (images + torch.randn_like(images) * self.base_noise_std).clamp(0, 1)


            with torch.no_grad():
                grabber.clear()
                _ = self.get_logits(x_base)                
                f_base = grabber.get(self.feature_layer).detach()
            basefeat_list.append(f_base)

            acc_w = None
            for l in range(max(self.ens, 1)):
                lam = l / max(self.ens, 1)
                x_mix = images + torch.randn_like(images) * self.mix_noise_std
                x_mix = (1 - lam) * x_mix + lam * x_base
                x_in = input_diversity(x_mix, self.resize_rate, self.diversity_prob) if self.diversity else x_mix

                grabber.clear()
                x_in.requires_grad_(True)
                logits = self.get_logits(x_in)
                loss = (logits * one_hot).sum()
                f = grabber.get(self.feature_layer)
                g = torch.autograd.grad(loss, f, retain_graph=False, create_graph=False, allow_unused=False)[0].detach()
                acc_w = g if acc_w is None else (acc_w + g)

            acc_w = -l2_normalize_per_sample(acc_w)  
            weights_list.append(acc_w)

        weights_stack = torch.stack(weights_list, dim=0)   # (N_bases,B,C,H,W)
        base_feats = torch.stack(basefeat_list, dim=0)     # (N_bases,B,C,H,W)
        return weights_stack, base_feats

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv = images.clone().detach()
        momentum = torch.zeros_like(images, device=self.device)
        amp = torch.zeros_like(images, device=self.device)  # PIM
        losses = []
        all_images = []

        grabber = FeatureGrabber(self.model, [self.feature_layer])

        weights_stack, base_feats = self._prepare_weights_and_bases(
            images, labels if not self.targeted else target_labels, grabber
        )  # (N,B,C,H,W)

        for _ in range(self.steps):
            adv.requires_grad_(True)
            grabber.clear()

            x_in = input_diversity(adv, self.resize_rate, self.diversity_prob) if self.diversity else adv
            logits = self.get_logits(x_in)

            f_adv = grabber.get(self.feature_layer)  # (B,C,H,W)
            # (N,B,C,H,W): (f_adv - f_base) * weights
            attrib = (f_adv.unsqueeze(0) - base_feats) * weights_stack
            attrib_sum = attrib.sum(dim=0)

            positive = torch.clamp(attrib_sum, min=0.0)
            negative = torch.clamp(attrib_sum, max=0.0)
            balance_attrib = positive + self.gamma_neg * negative

            loss = balance_attrib.mean()
            losses.append(loss.detach())

            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

            if self.TI:
                k = self.stacked_kernel.to(grad.device, dtype=grad.dtype)
                grad = F.conv2d(grad, k, stride=1, padding='same', groups=grad.shape[1])

            grad_n = mean_abs_normalize_per_sample(grad)
            momentum = self.decay * momentum + grad_n

            if self.PIM:
                alpha_beta = self.alpha * self.amp_factor
                amp = amp + alpha_beta * momentum.sign()
                cut_noise = (amp.abs() - self.eps).clamp(min=0.0) * amp.sign()
                pker = self.pim_kernel.to(grad.device, dtype=grad.dtype)
                projection = self.gamma_pim * F.conv2d(cut_noise, pker, stride=1, padding='same', groups=grad.shape[1])
                amp = amp + projection
                adv = adv.detach() + alpha_beta * momentum.sign() + projection
            else:
                adv = adv.detach() + self.alpha * momentum.sign()

            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv)

        grabber.close()
        return all_images, losses
