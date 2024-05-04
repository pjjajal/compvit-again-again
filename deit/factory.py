from typing import Literal

from .deit import (
    deit3_base_patch16_224,
    deit3_large_patch16_224,
    deit3_small_patch16_224,
)


def compdeit_factory(name: Literal["small", "base", "large"]):
    if name == "small":
        model, model_conf = deit3_small_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
        )
    elif name == "base":
        model, model_conf = deit3_base_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
        )
    elif name == "large":
        model, model_conf = deit3_large_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
        )
    return model, model_conf


def distill_factory(
    teacher_name: Literal["small", "base", "large"],
    student_name: Literal["small", "base", "large"],
    **kwargs
):

    if teacher_name == "small":
        teacher, teacher_conf = deit3_small_patch16_224(
            pretrained=True, dynamic_img_size=True
        )
    elif teacher_name == "base":
        teacher, teacher_conf = deit3_base_patch16_224(
            pretrained=True, dynamic_img_size=True
        )
    elif teacher_name == "large":
        teacher, teacher_conf = deit3_large_patch16_224(
            pretrained=True, dynamic_img_size=True
        )

    if student_name == "small":
        student, student_conf = deit3_small_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
        )
    elif student_name == "base":
        student, student_conf = deit3_base_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
        )
    elif student_name == "large":
        student, student_conf = deit3_large_patch16_224(
            pretrained=True,
            compvit=True,
            pretrained_strict=False,
            dynamic_img_size=True,
        )

    return (
        student,
        teacher,
        {
            **teacher_conf,
            **student_conf,
        },
    )
