import os
from typing import Literal
from pathlib import Path

from omegaconf import OmegaConf

from .deit import (
    deit_tiny_patch16_224,
    deit3_base_patch16_224,
    deit3_large_patch16_224,
    deit3_small_patch16_224,
)

CONFIG_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "configs"


def compdeit_factory(name: Literal["tiny", "small", "base", "large"], **kwargs):
    config_path = CONFIG_PATH / "compvit_deit.yaml"
    # Loads the default configuration.
    conf = OmegaConf.load(config_path)
    conf = conf[name]

    if name == "tiny":
        model, model_conf = deit_tiny_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
            **conf,
        )
    elif name == "small":
        model, model_conf = deit3_small_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
            **conf,
        )
    elif name == "base":
        model, model_conf = deit3_base_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
            **conf,
        )
    elif name == "large":
        model, model_conf = deit3_large_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
            **conf,
        )
    return model, model_conf


def distill_factory(
    teacher_name: Literal["tiny", "small", "base", "large"],
    student_name: Literal["tiny", "small", "base", "large"],
    **kwargs
):
    if teacher_name == "tiny":
        teacher, teacher_conf = deit_tiny_patch16_224(
            pretrained=True,
            dynamic_img_size=True,
        )
    elif teacher_name == "small":
        teacher, teacher_conf = deit3_small_patch16_224(
            pretrained=True,
            dynamic_img_size=True,
        )
    elif teacher_name == "base":
        teacher, teacher_conf = deit3_base_patch16_224(
            pretrained=True,
            dynamic_img_size=True,
        )
    elif teacher_name == "large":
        teacher, teacher_conf = deit3_large_patch16_224(
            pretrained=True,
            dynamic_img_size=True,
        )

    student, student_conf = compdeit_factory(student_name, **kwargs)

    return (
        student,
        teacher,
        {
            **teacher_conf,
            **student_conf,
        },
    )

if __name__ == "__main__":
    student, teacher, conf = distill_factory("small", "tiny")
    print(student)
    print(teacher)
    print(conf)