import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Union

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tvt

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from compvit.factory import compvit_factory
from deit.factory import compdeit_factory
from deit.deit import CompDeiT
from compvit.models.compvit import CompViT
from datasets import create_dataset
from dinov2.factory import dinov2_factory
from dinov2.models.vision_transformer import DinoVisionTransformer
from utils.schedulers import CosineAnnealingWithWarmup


CONFIG_PATH = Path("./configs")
DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    parser.add_argument("--model", choices=["dino", "deit"], default="dino")
    parser.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100", "imagenet"]
    )
    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--checkpoints_path", type=Path, default=None)
    parser.add_argument(
        "--precision",
        choices=[
            "32-true",
            "32",
            "16-mixed",
            "bf16-mixed",
            "transformer-engine",
            "16-true",
            "bf16-true",
            "64-true",
        ],
        default="bf16-mixed",
    )
    parser.add_argument(
        "--overfit_batches",
        type=float,
        default=0,
        help="Overfit on a subset of the data for debugging purposes",
    )
    parser.add_argument(
        "--use_mixup", default=False, action="store_true", help="Use mixup"
    )
    parser.add_argument(
        "--use_cutmix", default=False, action="store_true", help="Use cutmix"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--save_tag", type=str, default="")

    return parser.parse_args()


class LinearClassifierModel(nn.Module):
    def __init__(
        self,
        model: Union[CompDeiT, CompViT],
        num_classes,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(model, (CompDeiT, CompViT, DinoVisionTransformer))

        self.num_classes = num_classes
        self.model = model

        if isinstance(model, CompDeiT):
            self.head = nn.Linear(model.embed_dim, num_classes)
        elif isinstance(model, (CompViT, DinoVisionTransformer)):
            self.head = nn.Linear(model.embed_dim * 2, num_classes)

    def forward(self, x, **kwargs):
        if isinstance(self.model, CompDeiT):
            outputs = self.model(x, is_training=True)
            outputs = outputs["x_norm_clstoken"]
        else:
            outputs = self.model(x, is_training=True)
            cls_token = outputs["x_norm_clstoken"]
            patch_tokens = outputs["x_norm_patchtokens"]
            outputs = torch.cat(
                [
                    cls_token,
                    patch_tokens.mean(dim=1),
                ],
                dim=1,
            )
        outputs = self.head(outputs)
        return outputs


class LightningFT(L.LightningModule):
    def __init__(
        self,
        model: Union[LinearClassifierModel],
        args,
        hyperparameters,
    ) -> None:
        super().__init__()
        self.model = model
        self.hyperparameters = hyperparameters
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.900)
        )

        self.cutmix_or_mixup = []
        if args.use_mixup:
            self.cutmix_or_mixup.append(
                tvt.MixUp(
                    alpha=hyperparameters["mixup_alpha"],
                    num_classes=model.num_classes,
                )
            )
        if args.use_cutmix:
            self.cutmix_or_mixup.append(
                tvt.CutMix(
                    alpha=hyperparameters["mixup_alpha"],
                    num_classes=model.num_classes,
                )
            )
        if not self.cutmix_or_mixup:
            self.cutmix_or_mixup = [tvt.Identity()]
        self.cutmix_or_mixup = tvt.RandomChoice(self.cutmix_or_mixup)

        self.running_loss = 0
        self.highest_val_accuracy = float("-inf")
        self.accuracy_top1 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=1
        )
        self.accuracy_top5 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=5
        )

    def training_step(self, batch, batch_idx):
        x, label = batch
        x, label = self.cutmix_or_mixup(x, label)
        outputs = self.model(x)
        loss = self.criterion(outputs, label)
        # Running loss.
        self.running_loss += loss.detach().item()
        self.log("train loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        outputs = self.model(x)
        loss = self.criterion(outputs, label)

        # Accuracy
        self.accuracy_top1(outputs, label)
        self.accuracy_top5(outputs, label)
        self.log(
            "accuracy_top1",
            self.accuracy_top1,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "accuracy_top5",
            self.accuracy_top5,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("test loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparameters["lr"],
            weight_decay=5e-2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWithWarmup(
                    optimizer,
                    T_max=self.hyperparameters["epochs"],
                    eta_min=self.hyperparameters["min_lr"],
                    warmup_epochs=self.hyperparameters["warmup_epochs"],
                ),
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self) -> None:
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_model)

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure=None
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema_model.update_parameters(self.model)

    def on_validation_epoch_end(self) -> None:
        acc = self.accuracy_top1.compute()
        if acc > self.highest_val_accuracy:
            self.highest_val_accuracy = acc
            # Save Model
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"best_performing.pth"
                torch.save(self.model.state_dict(), save_path)


def main(args):
    config_path = CONFIG_PATH / (args.dataset + f"_{args.model}" + ".yaml")
    configs = OmegaConf.load(config_path)
    model_config = configs["compvit"]
    head_config = configs["head"]
    hyperparameters = configs["hyperparameters"]

    # Merging config with CLI args. CLI is prioritized over config.
    args = OmegaConf.merge(
        configs["args"],
        vars(args),
    )

    if args.model == "dino":
        model, config = compvit_factory(model_config["name"])
        if model_config["checkpoint"]:
            print("Loading", model_config["checkpoint"])
            model.load_state_dict(torch.load(model_config["checkpoint"]), strict=False)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint), strict=False)

    if args.model == "deit":
        model, config = compdeit_factory(model_config["name"])
        if model_config["checkpoint"]:
            print("Loading", model_config["checkpoint"])
            model.load_state_dict(torch.load(model_config["checkpoint"]), strict=False)
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint), strict=False)

    # Create classifier model.
    model = LinearClassifierModel(model, head_config["num_classes"])

    model = LightningFT(model, args, hyperparameters)

    # Setup W&B.
    wandb_logger = WandbLogger(project="compvit-again-again")

    # Create lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Create trainer.
    trainer = L.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=hyperparameters["accumulations"],
        max_epochs=hyperparameters["epochs"],
        logger=wandb_logger,
        benchmark=True,  # cudnn benchmarking, allows for faster training.
        enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
        callbacks=[lr_monitor],
        overfit_batches=args.overfit_batches,
        log_every_n_steps=50,
        strategy="ddp_find_unused_parameters_true",
    )

    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(
            {
                **config,
                **hyperparameters,
                **args,
            }
        )

    # Create dataset and train loader.
    train_dataset, test_dataset = create_dataset(args)
    loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False if args.overfit_batches else True,
        num_workers=hyperparameters["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparameters["test_batch_size"],
        shuffle=False,
        # num_workers=2,
        num_workers=hyperparameters["num_workers"],
    )
    # Trainer Fit.
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    args = parse_args()

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if args.save_tag:
        now = now + args.save_tag

    save_loc = DEFAULT_CHECKPOINTS_PATH / now
    if args.checkpoints_path:
        save_loc = args.checkpoints_path / now

    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = False
    main(args)
