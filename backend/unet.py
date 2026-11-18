"""Pixel land-cover model used by the website.

This is separate from the thesis ResNet-18 patch classifier.
It labels every pixel with one of the five Wuhan classes.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


CLASS_NAMES = [
    "Tree cover",
    "Grassland",
    "Cropland",
    "Built-up",
    "Permanent water",
]

ARCHITECTURE = "small_unet_landcover"


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):

    def __init__(self, num_classes=5, mean=None, std=None):
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

        if mean is None:
            mean = [0.0, 0.0, 0.0]
        if std is None:
            std = [1.0, 1.0, 1.0]

        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        bottleneck = self.enc3(self.pool(skip2))
        decoded = self.up2(bottleneck)
        decoded = self.dec2(torch.cat([decoded, skip2], dim=1))
        decoded = self.up1(decoded)
        decoded = self.dec1(torch.cat([decoded, skip1], dim=1))
        return self.head(decoded)


def load_unet(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SmallUNet(
        num_classes=len(checkpoint.get("class_names", CLASS_NAMES)),
        mean=checkpoint["normalization_mean"],
        std=checkpoint["normalization_std"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _window_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.inference_mode()
def segment_image(model, image_chw, device, tile=256, stride=192):
    """Return class indices and the mean softmax of each class.

    image_chw is float32 RGB in channel-first order. Non-finite pixels are
    ignored and marked with class index -1.
    """

    image = np.asarray(image_chw, dtype=np.float32)
    valid = np.isfinite(image).all(axis=0)
    filled = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    channels, height, width = filled.shape

    accumulator = np.zeros((model.head.out_channels, height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)

    for y0 in _window_starts(height, tile, stride):
        for x0 in _window_starts(width, tile, stride):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            crop = filled[:, y0:y1, x0:x1]
            crop_h, crop_w = crop.shape[1:]
            pad_h = (4 - crop_h % 4) % 4
            pad_w = (4 - crop_w % 4) % 4
            if pad_h or pad_w:
                crop = np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)))

            tensor = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0).to(device)
            logits = model(tensor)[0, :, :crop_h, :crop_w].cpu().numpy()
            accumulator[:, y0:y1, x0:x1] += logits
            coverage[y0:y1, x0:x1] += 1.0

    coverage = np.maximum(coverage, 1.0)
    logits = accumulator / coverage
    probabilities = np.exp(logits - logits.max(axis=0, keepdims=True))
    probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-8)

    prediction = probabilities.argmax(axis=0).astype(np.int16)
    prediction[~valid] = -1

    valid_count = int(valid.sum())
    if valid_count == 0:
        mean_probabilities = np.zeros(probabilities.shape[0], dtype=np.float32)
    else:
        mean_probabilities = probabilities[:, valid].mean(axis=1)

    return prediction, mean_probabilities, valid
