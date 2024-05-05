import torch
from torchmetrics import Accuracy

from compvit.factory import compvit_factory
from dinov2.factory import dinov2_factory
from exited_models.patch import exit_patch
from thirdparty.tome.factory import dinov2_tome_factory

from datasets import create_dataset
from dataclasses import dataclass
from omegaconf import OmegaConf
from train_ft import LinearClassifierModel
from tqdm import tqdm
import pandas as pd

@dataclass
class EvalConfig:
    dataset: str = "imagenet"
    data_dir: str = "~/datasets/imagenet"
    pretraining: bool = False
    batch_size: int = 64


def eval_model(model, dataset, device, batch_size=32):
    model.eval()

    accuracy_top1 = Accuracy("multiclass", num_classes=1000, top_k=1).to(device)
    accuracy_top5 = Accuracy("multiclass", num_classes=1000, top_k=5).to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            accuracy_top1(outputs, labels)
            accuracy_top5(outputs, labels)

    return accuracy_top1.compute(), accuracy_top5.compute()


def main():
    eval_config = OmegaConf.structured(EvalConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = compvit_factory("vit_base_patch16_224", pretrained=True)
    # model = dinov2_factory("dinov2_s", pretrained=True)
    # model = exit_patch("vit_base_patch16_224", pretrained=True)

    data = []
    for r in range(2, 25):
        model, config = dinov2_tome_factory("dinov2_vits14", r=r)

        model = LinearClassifierModel(model, 1000)
        model.model.load_state_dict(
            torch.load("dinov2/checkpoints/dinov2_vits14_pretrain.pth")
        )
        model.head.load_state_dict(
            torch.load("dinov2/checkpoints/dinov2_vits14_linear_head.pth")
        )
        model = model.to(device)

        train_dataset, eval_dataset = create_dataset(eval_config)

        top1, top5 = eval_model(model, eval_dataset, device, batch_size=eval_config.batch_size)
        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}")
        data.append({"r": r,"top1": 100 * top1.item(), "top5": 100 * top5.item()})
    
    pd.DataFrame(data).to_csv("benchmarks/eval_results/dinov2_vits14_tome.csv")


if __name__ == "__main__":
    main()
