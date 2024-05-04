from typing import Sequence

import json
import torch
import torchvision.transforms.v2 as tvt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageNet


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)


def create_imagenet_dataset(args):

    if args.pretraining:
        train_transform = tvt.Compose(
            [
                tvt.ToImage(),
                tvt.ToDtype(torch.float32, scale=True),
                tvt.RandomResizedCrop(224),
                tvt.RandomHorizontalFlip(),
                # make_normalize_transform(),
            ]
        )
    else:
        train_transform = tvt.Compose(
            [
                tvt.RandomResizedCrop(224),
                tvt.RandomHorizontalFlip(),
                tvt.RandomChoice(
                    [
                        tvt.GaussianBlur(7),
                        tvt.RandomSolarize(threshold=0.5, p=1),
                        tvt.RandomGrayscale(p=1),
                    ]
                ),
                tvt.ColorJitter(0.3, 0.3, 0.3, 0.2),
                tvt.ToTensor(),
                make_normalize_transform(),
            ]
        )
    val_transform = tvt.Compose(
        [
            tvt.Resize(256, interpolation=tvt.InterpolationMode.BICUBIC),
            tvt.CenterCrop(224),
            tvt.ToTensor(),
            make_normalize_transform(),
        ]
    )

    train_dataset = ImageNet(args.data_dir, "train", transform=train_transform)
    val_dataset = ImageNet(args.data_dir, "val", transform=val_transform)

    return train_dataset, val_dataset
