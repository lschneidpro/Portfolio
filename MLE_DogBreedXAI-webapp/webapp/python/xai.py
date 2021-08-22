#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:47:13 2020

@author: luca
"""

import base64
from io import BytesIO
import json

import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms


PATH_JSON = "python/breeds.txt"
PATH_MODEL = "python/ft_densenet121.pt"


def b64_to_pil(string):

    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im = Image.open(buffer)

    return im


def pil_to_b64(pil_img):

    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()

    return img_str


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def preprocess(pil_img):
    raw_image = pil_to_cv2(pil_img)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def get_gradcam(gcam, raw_image):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return np.uint8(gcam)


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def get_xai(pil_img):

    # Preprocess Image
    pt_img, cv2_raw_img = preprocess(pil_img)
    pt_img = torch.stack([pt_img]).to(device)

    # Run forward() with images
    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(pt_img)

    # Run backward() with a list of specific classes
    gcam.backward(ids=ids[:, [0]])
    regions = gcam.generate(target_layer="features")
    gcam.remove_hook()

    # Run Grad-Cam image
    cv2_grad_img = get_gradcam(regions[0, 0], cv2_raw_img)

    # Export Images
    resized_grad_img = cv2.resize(
        cv2_grad_img, (400, 400), interpolation=cv2.INTER_CUBIC
    )
    pil_grad_img = cv2_to_pil(resized_grad_img)
    resized_raw_img = cv2.resize(cv2_raw_img, (400, 400), interpolation=cv2.INTER_CUBIC)
    pil_raw_img = cv2_to_pil(resized_raw_img)

    # Export top 5 breeds
    breed_5 = [list_breed[i] for i in ids[:, 0:5].numpy()[0]]
    probs_5 = probs[:, 0:5].detach().numpy()[0]
    d_probs = {"breed": breed_5, "probability": probs_5}

    return pil_raw_img, pil_grad_img, d_probs


# main
device = get_device(False)


# Get Breeds list
with open(PATH_JSON, "r") as filehandle:
    list_breed = json.load(filehandle)
n_classes = len(list_breed)


# Prepare fine-tuned Model
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, n_classes)
model.load_state_dict(torch.load(PATH_MODEL, map_location=torch.device(device)))
model.to(device)
model.eval()
