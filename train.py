import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tvt
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy

from compvit.layers.projection import Projection
from compvit.models.compvit import CompViT
from datasets import create_dataset
from datasets.imagenet21k.augment import Augmentation
from dinov2.models.vision_transformer import DinoVisionTransformer
from utils.schedulers import CosineAnnealingWithWarmup

CONFIG_PATH = Path("./configs")
DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")
    # General Arguments.
    parser.add_argument("--model", choices=["dino", "deit"], default="dino")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "cifar100", "imagenet", "imagenet-21k"],
    )

    # Trainer Specific Arguments.
    parser.add_argument("--downsize", type=int, default=224)
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

    # Distillation arguments.
    parser.add_argument("--kl", default=False, action="store_true", help="Use KL loss")
    parser.add_argument("--temp", type=float, default=1, help="KL temp")
    parser.add_argument(
        "--augmentations", default=False, action="store_true", help="Use augmentations"
    )
    parser.add_argument(
        "--rand-aug", default=False, action="store_true", help="Use RandAugment"
    )
    parser.add_argument(
        "--symmetric", default=False, action="store_true", help="Use symmetric downsize"
    )
    parser.add_argument(
        "--use_mixup", default=False, action="store_true", help="Use mixup"
    )
    parser.add_argument(
        "--use_cutmix", default=False, action="store_true", help="Use cutmix"
    )
    parser.add_argument("--ema", default=0.98)
    # Other arguments.
    parser.add_argument(
        "--cache_save_path", type=Path, default=None, help="Path where to cache data"
    )
    parser.add_argument(
        "--checkpoint_per_epoch",
        default=False,
        action="store_true",
        help="Enable to checkpoint per epoch",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Use a subset of the data for training",
    )

    return parser.parse_args()


class LightningDistill(L.LightningModule):
    def __init__(
        self,
        student: CompViT,
        teacher: DinoVisionTransformer,
        args,
        hyperparameters,
        config,
    ):
        super().__init__()
        # Args, hyperparameters, and config.
        self.args = args
        self.hyperparameters = hyperparameters
        self.config = config

        # Student and teacher models.
        self.student = student
        self.teacher = teacher

        self.ema_student = torch.optim.swa_utils.AveragedModel(
            self.student, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.args.ema)
        )

        # Decoder.
        if self.args.model == "dino":
            self.proj = Projection(student.embed_dim * 2, teacher.embed_dim * 2, normalize=False)
        elif self.args.model == "deit":
            self.proj = Projection(student.embed_dim, teacher.embed_dim , normalize=False)

        self.head = nn.Linear(student.embed_dim * 2, hyperparameters["mixup_classes"])
        self.criterion = nn.CrossEntropyLoss()

        # Transformations.
        self.downsize = tvt.Resize(args.downsize)

        self.cutmix_or_mixup = []
        if args.use_mixup:
            self.cutmix_or_mixup.append(
                tvt.MixUp(
                    alpha=hyperparameters["mixup_alpha"],
                    num_classes=hyperparameters["mixup_classes"],
                )
            )
        if args.use_cutmix:
            self.cutmix_or_mixup.append(
                tvt.CutMix(
                    alpha=hyperparameters["mixup_alpha"],
                    num_classes=hyperparameters["mixup_classes"],
                )
            )
        self.cutmix_or_mixup = tvt.RandomChoice(self.cutmix_or_mixup)

        if args.augmentations:
            self.augment = Augmentation()
        if args.rand_aug:
            self.augment = tvt.RandAugment(num_ops=3, magnitude=10)

        # Loss tracking.
        self.running_loss = 0
        self.lowest_batch_loss = float("inf")

        self.highest_val_accuracy = float("-inf")
        self.accuracy_top1 = Accuracy(
            "multiclass", num_classes=hyperparameters["mixup_classes"], top_k=1
        )
        self.accuracy_top5 = Accuracy(
            "multiclass", num_classes=hyperparameters["mixup_classes"], top_k=5
        )

    @torch.no_grad()
    def forward_teacher(self, x):
        x = self.teacher.forward(x, is_training=True)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"].mean(dim=1)
        return cls_token, patch_tokens

    def forward_student(self, x):
        x = self.student.forward(x, is_training=True)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"].mean(dim=1)
        return cls_token, patch_tokens

    def mse_loss(self, x, y, alpha=2):
        return F.mse_loss(x, y)

    def calculate_loss(self, x, x_teacher):
        if self.args.kl:
            return F.kl_div(
                F.log_softmax(x, dim=-1),
                F.softmax(x_teacher * self.args.temp, dim=-1),
                reduction="batchmean",
            )
        
        if self.args.model == "dino":
            x = F.layer_norm(x, (self.teacher.embed_dim * 2,))
            x_teacher = F.layer_norm(x_teacher, (self.teacher.embed_dim * 2,))
        else:
            x = F.layer_norm(x, (self.teacher.embed_dim,))
            x_teacher = F.layer_norm(x_teacher, (self.teacher.embed_dim,))
        return self.mse_loss(
            x, x_teacher
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        resize_op = self.downsize

        x = self.augment(x) if self.args.augmentations else x
        if self.args.use_mixup or args.use_cutmix:
            x, y = self.cutmix_or_mixup(x, y)

        # Teacher forward.
        teacher_cls, teacher_patch = self.forward_teacher(resize_op(x))

        # Student forward.
        student_cls, student_patch = self.forward_student(
            resize_op(x) if self.args.symmetric else x
        )

        if self.args.model == "dino":
            student_cls_embeddings = self.proj(torch.cat([student_cls, student_patch], dim=-1))
            teacher_cls = torch.cat([teacher_cls, teacher_patch], dim=-1)
        else:
            student_cls_embeddings = self.proj(student_cls)
        # student_cls_embeddings = self.proj(student_cls)
        # student_patch_embeddings = self.proj_patch(student_patch)

        # Loss.
        loss_cls = self.calculate_loss(student_cls_embeddings, teacher_cls)

        # loss_patch = 0
        # if not self.args.model == "deit":
        #     loss_patch = self.calculate_loss(student_patch_embeddings, teacher_patch)

        loss = loss_cls

        # Running loss.
        self.running_loss += loss.detach().item()
        self.log(
            "cls loss",
            loss_cls,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # self.log(
        #     "patch loss",
        #     loss_patch,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )
        self.log(
            "train loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema_student.update_parameters(self.student)

    def configure_optimizers(self):
        # Determine parameters to optimize.
        parameters = list(self.student.parameters())
        parameters += list(self.proj.parameters())  # Add decoder parameters.
        # parameters += list(self.proj_patch.parameters())
        optimizer = optim.AdamW(
            parameters,
            lr=self.hyperparameters["lr"],
            weight_decay=1e-4,
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
        torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.ema_student)
        if self.args.checkpoint_per_epoch:
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"model_epoch_{self.current_epoch}.pth"
                torch.save(self.student.state_dict(), save_path)

        if self.running_loss < self.lowest_batch_loss:
            self.lowest_batch_loss = self.running_loss
            self.running_loss = 0
            # Save Model
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"best_performing.pth"
                torch.save(self.student.state_dict(), save_path)
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"best_performing_ema.pth"
                torch.save(self.ema_student.state_dict(), save_path)


def main(args):
    config_path = CONFIG_PATH / (args.dataset + f"_pt_{args.model}" + ".yaml")
    configs = OmegaConf.load(config_path)
    teacher_config = configs["teacher"]
    student_config = configs["student"]
    hyperparameters = configs["hyperparameters"]

    # Merging config with CLI args. CLI is prioritized over config.
    args = OmegaConf.merge(
        configs["args"],
        vars(args),
    )

    # Get checkpoint paths.
    teacher_checkpoint = teacher_config["checkpoint"]
    student_checkpoint = student_config["checkpoint"]

    # Create models.
    if args.model == "deit":
        from deit.factory import distill_factory
    elif args.model == "dino":
        from compvit.factory import distill_factory

    student, teacher, config = distill_factory(
        teacher_name=teacher_config["name"],
        student_name=student_config["name"],
    )
    if teacher_checkpoint:
        teacher.load_state_dict(torch.load(teacher_checkpoint))
    if student_checkpoint:
        student.load_state_dict(torch.load(student_checkpoint), strict=False)

    model = LightningDistill(student, teacher, args, hyperparameters, config)

    # Setup W&B.
    wandb_logger = WandbLogger(project="compvit-again-again", prefix="distill")

    # # Create lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # # Create trainer.
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
                "architecture": "mae",
                "dataset": args.dataset,
                "teacher": teacher_config["name"],
                "student": student_config["name"],
                "teacher_checkpoint": teacher_config["checkpoint"],
                "student_checkpoint": student_config["checkpoint"],
                **config,
                **hyperparameters,
                **args,
            }
        )

    # Create dataset and train loader.
    train_dataset, test_dataset = create_dataset(args)

    # Cache data for imagnet-21k.
    if args.cache_save_path and args.dataset == "imagenet-21k":
        cache_data = train_dataset.cache_data()
        args.cache_save_path.mkdir(parents=True, exist_ok=True)
        save_path = args.cache_save_path / f"{args.dataset}_cache_data.json"
        with open(save_path, "w") as f:
            json.dump(cache_data, f)

    if args.subset:
        if trainer.global_rank == 0:
            wandb_logger.experiment.config.update(
                {
                    "subset_percentage": args.subset,
                    "subset": int(len(train_dataset) * args.subset),
                    "total_size": len(train_dataset),
                },
                allow_val_change=True,
            )
        train_dataset = Subset(
            train_dataset,
            torch.randperm(len(train_dataset))[: int(len(train_dataset) * args.subset)],
        )

    # Create train loader.
    loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False if args.overfit_batches else True,
        num_workers=hyperparameters["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    # Trainer Fit.
    trainer.fit(model, train_dataloaders=loader)


if __name__ == "__main__":
    args = parse_args()

    now = "distill_" + datetime.now().strftime("%Y-%m-%d-%H%M%S")

    save_loc = DEFAULT_CHECKPOINTS_PATH / now
    if args.checkpoints_path:
        save_loc = args.checkpoints_path / now

    if not save_loc.exists():
        save_loc.mkdir(parents=True, exist_ok=True)

    args.save_loc = save_loc
    args.pretraining = True
    main(args)
