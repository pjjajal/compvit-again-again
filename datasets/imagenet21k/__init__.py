import json
import torch
import torchvision.transforms.v2 as tvt
from torchvision.datasets import ImageFolder
from utils.cfolder import CachedImageFolder
from .augment import make_normalize_transform


def create_imagenet21k_dataset(args):
    # if args.augmentations:
    #     transform = tvt.Compose(
    #         [
    #             tvt.ToImage(),
    #             tvt.ToDtype(torch.float32, scale=True),
    #             tvt.RandomResizedCrop(224),
    #             tvt.RandomHorizontalFlip(),
    #             tvt.RandomChoice(
    #                 [
    #                     tvt.GaussianBlur(7),
    #                     tvt.RandomSolarize(threshold=0.5, p=1),
    #                     tvt.RandomGrayscale(p=1),
    #                 ]
    #             ),
    #             make_normalize_transform(),
    #         ]
    #     )
    # else:
    transform = tvt.Compose(
        [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(224),
            tvt.RandomHorizontalFlip(),
            # make_normalize_transform(),
        ]
    )

    cached_data = None
    if args.cache_path:
        with open(args.cache_path, "r") as f:
            cached_data = json.load(f)

    return CachedImageFolder(
        root=args.data_dir, transform=transform, cached_data=cached_data
    )

    # return ImageFolder(root=args.data_dir, transform=transform)
