#添加路径
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats as st

from torchattacks.attack import Attack
import math
from load_dm import get_imagenet_dm_conf
import warnings

# 忽略所有的 UserWarning 和 FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class FGSM(Attack):
    r"""
    Iterative FGSM

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, steps = 10, resize_rate=0.9, diversity_prob=0.5,
                kernel_name='gaussian', len_kernel=15, nsig=3, diversity=False,
                TI=False, SI = False, m = 5, alpha=2/255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.nsig = nsig
        self.len_kernel = len_kernel
        self.diversity = diversity
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.m = m
        self.SI = SI
        self.diversity = diversity
        self.TI = TI
        self.steps = steps
        self.alpha = alpha

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        losses = []
        all_images = []

        stacked_kernel = self.stacked_kernel.to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            if self.SI:
                adv_grad = torch.zeros_like(images).detach().to(self.device)
                cost1 = 0
                for i in torch.arange(self.m):
                    nes_images = adv_images / torch.pow(2, i)
                    if self.diversity:
                        outputs = self.get_logits(self.input_diversity(nes_images))
                    else:
                        outputs = self.get_logits(nes_images)
                    # Calculate loss
                    if self.targeted:
                        cost = -loss(outputs, target_labels)
                    else:
                        cost = loss(outputs, labels)
                    cost1 = cost1 + cost
                    adv_grad += torch.autograd.grad(
                        cost, adv_images, retain_graph=False, create_graph=False
                    )[0]
                losses.append(cost1/self.m)
                grad = adv_grad / self.m
            else:
                if self.diversity:
                    outputs = self.get_logits(self.input_diversity(adv_images))
                else:
                    outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                losses.append(cost)
                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

            if self.TI:
                # depth wise conv2d
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            all_images.append(adv_images)

        return all_images,losses

class NIFGSM(Attack):
    r"""
    NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.NIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        super().__init__("NIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        losses = []
        all_images = []
        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum
            outputs = self.get_logits(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            losses.append(cost)
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad = self.decay * momentum + grad / torch.mean(
                torch.abs(grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv_images)

        return all_images,losses

class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, m=5):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        losses = []
        all_images = []

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images + self.decay * self.alpha * momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().to(self.device)
            cost_1 = 0
            for i in torch.arange(self.m):
                nes_images = nes_image / torch.pow(2, i)
                outputs = self.get_logits(nes_images)
                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                cost_1 = cost_1 + cost
                adv_grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]
            losses.append(cost_1/self.m)
            adv_grad = adv_grad / self.m

            # Update adversarial images
            grad = self.decay * momentum + adv_grad / torch.mean(
                torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv_images)

        return all_images,losses

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        losses = []
        all_images = []
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            losses.append(cost)
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv_images)

        return all_images,losses

class diff_PGD(Attack):
    r"""

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, respace = 'ddim50',t = 1):
        super().__init__("diff_PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.t = t
        self.diffmodel, self.diffusion = get_imagenet_dm_conf(device=self.device, respace=respace)
        self.net = SDEdit(self.diffusion, self.diffmodel, t=self.t)

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        
        losses = []
        all_images = []
        for _ in range(self.steps):

            with torch.no_grad():
                adv_images_diff = self.net.sdedit(adv_images, self.t).detach()
            adv_images_diff.requires_grad = True
            # adv_images.requires_grad = True
            outputs = self.get_logits(adv_images_diff)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            losses.append(cost)
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images_diff, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv_images)
        
        
        # with torch.no_grad():
        #     pred_x0 = self.net.sdedit(adv_images, self.t)
        
        return all_images, losses

        # return adv_images, pred_x0

class AdaMSIFGM(Attack):
    r"""


    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 4/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AdaMSIFGM(model, eps=4/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=4/255, alpha=0.1/255, steps=10, delta=1e-16,
                 diversity=False, resize_rate=0.9, diversity_prob=0.5,
                 TI=False, kernel_name='gaussian', len_kernel=15, nsig=3):
        super().__init__("AdaMSIFGM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.delta = delta
        self.diversity = diversity
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.TI = TI
        self.supported_mode = ['default', 'targeted']

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        V_t = torch.zeros_like(images).detach().to(self.device)

        s_t_sub = 0

        s_t = 0

        gamma = 1

        lamb = 0.6

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        adv_images_last = images.clone().detach()
        stacked_kernel = self.stacked_kernel.to(self.device)

        losses = []
        all_images = []

        for i in range(self.steps):
            step = i + 1
            beta2_t = (step - gamma) / step
            xi_t = self.delta / (step ** 0.5)
            adv_images.requires_grad = True

            if self.diversity:
                outputs = self.get_logits(self.input_diversity(adv_images))
            else:
                outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            losses.append(cost)
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            if self.TI:
                # depth wise conv2d
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            V_t = beta2_t * V_t + (1 - beta2_t) * torch.mul(grad, grad)

            V_t_hat = torch.sqrt(V_t) + xi_t

            s_t = (torch.norm(grad, p=1) ** 1) * (lamb ** (step * 0.5))

            alpha_t = self.alpha

            if i == 0:
                s_t_sub = s_t + 1

            beta1_t = s_t_sub / (s_t + 1)

            s_t_sub = (torch.norm(grad, p=1) ** 1) * (lamb ** (step * 0.5))

            adv_images_next = adv_images.detach() + alpha_t * torch.div(grad, V_t_hat) + beta1_t * (adv_images.detach() - adv_images_last.detach())
            adv_images_last = adv_images
            delta = torch.clamp(adv_images_next - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            all_images.append(adv_images)

        return all_images, losses


        
class diff_AdaNAG(Attack):
    r"""
    
    
    Distance Measure : Linf
    
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 4/255)
    
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    
    Examples::
        >>> attack = diff_AdaNAG(model, eps=4/255)
        >>> adv_images = attack(images, labels)
    
    """
    def __init__(self, model, eps=4/255, eta=0.01, theta = 1, steps=10, 
                 delta=1e-16, SI = False, m = 5,
                 diversity=False, resize_rate=0.9, diversity_prob=0.5,
                 TI=False, kernel_name='gaussian', len_kernel=15, nsig=3,
                 respace = 'ddim50',t = 1):
        super().__init__("diff_AdaNAG_v1", model)
        self.eps = eps
        # self.alpha = alpha
        self.eta = eta
        self.theta = theta
        self.steps = steps
        self.delta = delta
        self.diversity = diversity
        self.SI = SI
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.TI = TI
        self.supported_mode = ['default', 'targeted']
        self.t = t
        self.diffmodel, self.diffusion = get_imagenet_dm_conf(device=self.device, respace=respace)
        self.net = SDEdit(self.diffusion, self.diffmodel, t=self.t)
        self.m = m
        

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        V_t = torch.zeros_like(images).detach().to(self.device)

        theta = self.theta

        # eta = 0.1
        # eta_t = 0.1
        # alpha = 200

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        z_t = images.clone().detach()
        adv_images_last = images.clone().detach()
        stacked_kernel = self.stacked_kernel.to(self.device)
        losses = []
        all_images = []

        # with torch.no_grad():
        #     images_diff = self.net.sdedit(images, 1).detach()
        # adv_images = images_diff.clone().detach()
        # z_t = images_diff.clone().detach()

        for j in range(self.steps):

            yt = (1 - theta) * adv_images + theta * z_t


            ##diffusion
            with torch.no_grad():
                yt_diff = self.net.sdedit(yt, self.t).detach()
            yt_diff.requires_grad = True
            # yt.requires_grad = True

            if self.SI:
                adv_grad = torch.zeros_like(images).detach().to(self.device)
                cost1 = 0
                for i in torch.arange(self.m):
                    yt_diffs = yt_diff / torch.pow(2, i)
                    if self.diversity:
                        outputs = self.get_logits(self.input_diversity(yt_diffs))
                    else:
                        outputs = self.get_logits(yt_diffs)
                    # Calculate loss
                    if self.targeted:
                        cost = -loss(outputs, target_labels)
                    else:
                        cost = loss(outputs, labels)
                    cost1 = cost1+cost
                    adv_grad += torch.autograd.grad(
                        cost, yt_diff, retain_graph=False, create_graph=False
                    )[0]
                grad = adv_grad / self.m
                losses.append(cost1/self.m)
            else:
                if self.diversity:
                    outputs = self.get_logits(self.input_diversity(yt_diff))
                else:
                    outputs = self.get_logits(yt_diff)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                losses.append(cost)
                # Update adversarial images
                grad = torch.autograd.grad(cost, yt_diff,
                                           retain_graph=False, create_graph=False)[0]


            if self.TI:
                # depth wise conv2d
                grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)

            # V_t = 0.9 * V_t + 0.1 * torch.mul(grad, grad) + self.delta
            V_t = V_t + torch.mul(grad, grad) + self.delta

            V_t_hat = torch.sqrt(V_t) #+ self.delta

            # 1
            eta_t = self.eta / (j + 2)

            
            z_t = z_t + eta_t * torch.div(grad, theta * V_t_hat)

            
            #1
            Delta = torch.clamp(z_t - images, min=-self.eps, max=self.eps)
            z_t = torch.clamp(images + Delta, min=0, max=1).detach()
            
            
            adv_images = (1 - theta) * adv_images.detach() + theta * z_t
            all_images.append(adv_images)
            theta = ( (theta**4 + 4 * theta)**(0.5) - theta**2 )/2

            # l_inf_norm = torch.norm(adv_images - images, p=float('inf'))
            # print(f"L∞ 范数: {l_inf_norm.item()}")
            

        # with torch.no_grad():
        #     pred_x0 = self.net.sdedit(adv_images, self.t)
        return all_images, losses


class SDEdit:
    def __init__(self, diffusion, model, t):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.t = t
    def sdedit(self, x, t, to_01=True):

            # assume the input is 0-1
            t_int = t
            
            x = x * 2 - 1
            
            t = torch.full((x.shape[0], ), t).long().to(x.device)
        
            x_t = self.diffusion.q_sample(x, t) 
            
            sample = x_t
        
            # print(x_t.min(), x_t.max())
        
            # si(x_t, 'vis/noised_x.png', to_01=True)
            
            indices = list(range(t_int+1))[::-1]

            
            # visualize 
            l_sample=[]
            l_predxstart=[]

            for i in indices:

                # out = self.diffusion.ddim_sample(self.model, sample, torch.tensor([i]).long().to(x.device))           
                out = self.diffusion.ddim_sample(self.model, sample, torch.full((x.shape[0], ), i).long().to(x.device))


                sample = out["sample"]


                l_sample.append(out['sample'])
                l_predxstart.append(out['pred_xstart'])
            
            

            # the output of diffusion model is [-1, 1], should be transformed to [0, 1]
            if to_01:
                sample = (sample + 1) / 2
            
            return sample
