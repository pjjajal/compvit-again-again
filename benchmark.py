# This code is awful. I'm sorry, but I will clean it up soon.
import warnings

### Ignore pesky warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

import argparse
import itertools
import textwrap
from functools import partial
from typing import Any, Dict, List, Literal

import pandas as pd
import torch
import torch.utils.benchmark as bench

from deit.deit import (
    deit3_base_patch16_224,
    deit_tiny_patch16_224,
    deit3_small_patch16_224,
    deit3_large_patch16_224,
)
from deit.factory import compdeit_factory
from compvit.factory import compvit_factory
from dinov2.factory import dinov2_factory
from exited_models.patch import exit_patch
from thirdparty.tome.factory import dinov2_tome_factory
from thirdparty.topk.factory import dinov2_topk_factory
from thirdparty.tome.patch.timm import apply_patch


def parse_args():
    parser = argparse.ArgumentParser("Benchmarking Code")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])
    single_group = parser.add_argument_group(
        title="Benchmark Single",
        description="Use these arguements to benchmark a SINGLE model.",
    )
    single_group.add_argument(
        "--model",
        choices=[
            "deit_tiny",
            "deit_small",
            "deit_base",
            "deit_large",
            "compdeit_tiny",
            "compdeit_small",
            "compdeit_base",
            "compdeit_large",
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
            "compvits14",
            "compvitb14",
            "compvitl14",
            "compvitg14",
        ],
    )
    single_group.add_argument("--wrap-tome", action="store_true")
    single_group.add_argument("--tome-r", type=int, default=0)

    benchmark_all_group = parser.add_argument_group(
        title="Benchmark All",
        description="Use these flags to benchmark a class of models.",
    )
    benchmark_all_group.add_argument("--all-compvit", action="store_true")
    benchmark_all_group.add_argument("--all-dino", action="store_true")
    benchmark_all_group.add_argument("--all-deit", action="store_true")
    benchmark_all_group.add_argument("--filetag", default="", type=str)

    sweep_group = parser.add_argument_group(
        title="CompViT Sweep", description="Use these arguments to ablate CompViT."
    )
    sweep_group.add_argument(
        "--sweep-model",
        choices=[
            "compvits14",
            "compvitb14",
            "compvitl14",
            "compvitg14",
        ],
    )
    sweep_group.add_argument("--compvit-sweep", action="store_true")
    sweep_group.add_argument(
        "--compressed-tokens-sweep", nargs="+", type=int, dest="token_sweep"
    )
    sweep_group.add_argument(
        "--bottleneck-loc-sweep", nargs=2, type=int, dest="bottleneck_locs"
    )

    tome_sweep = parser.add_argument_group(
        title="Tome Sweep", description="Use these arguments to ablate Tome."
    )
    tome_sweep.add_argument("--tome-sweep", action="store_true")
    tome_sweep.add_argument(
        "--tome-sweep-model",
        choices=[
            "deit_tiny",
            "deit_small",
            "deit_base",
            "deit_large",
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
    )

    tome_sweep = parser.add_argument_group(
        title="TopK Sweep", description="Use these arguments to ablate TopK."
    )
    tome_sweep.add_argument("--topk-sweep", action="store_true")
    tome_sweep.add_argument(
        "--topk-sweep-model",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )

    exiting = parser.add_argument_group(
        "Exiting Models", description="use these arguments to exit models early."
    )
    exiting.add_argument("--exit-sweep-dino", action="store_true")
    exiting.add_argument("--exit-sweep-tome", action="store_true")
    exiting.add_argument(
        "--exit-sweep-model",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    exiting.add_argument("--exit-sweep-r", type=int, default=2)

    return parser.parse_args()


def colour_text(
    text,
    colour_code: Literal[
        "black",
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "white",
        "reset",
    ],
    *args,
    **kwargs,
):
    colour_codes = {
        "black": "\033[90m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    coloured_text = colour_codes[colour_code] + str(text) + colour_codes["reset"]
    return coloured_text


def device_info(args):
    device = torch.device(args.device)
    device_name = ""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    return device_name


def compvit_message(
    model_name, config, model, latency_mean, latency_median, latency_iqr, final_tokens
):
    message = f"""\
    ========================
    {colour_text(model_name.upper(), 'green')}
    {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
    {colour_text("Depth", 'cyan')}: {config['depth']}
    {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
    {colour_text("num_compressed_tokens", 'cyan')}: {config['num_compressed_tokens']}
    {colour_text("bottleneck_loc", 'cyan')}: {config['bottleneck_loc']}
    {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f}
    {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
    {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
    {colour_text("Final Tokens", "magenta")}: {final_tokens}
    ========================\
    """
    return textwrap.dedent(message)


def baseline_message(
    model_name, config, model, latency_mean, latency_median, latency_iqr, final_tokens
):
    message = f"""\
        ========================
        {colour_text(model_name.upper(), 'green')}
        {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
        {colour_text("Depth", 'cyan')}: {config['depth']}
        {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
        {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f} 
        {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
        {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
        {colour_text("Final Tokens", "magenta")}: {final_tokens}
        ========================\
        """
    return textwrap.dedent(message)


def tome_message(
    model_name,
    config,
    model,
    latency_mean,
    latency_median,
    latency_iqr,
    final_tokens,
    r,
):
    message = f"""\
    ========================
    {colour_text(model_name.upper(), 'green')}
    {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
    {colour_text("Depth", 'cyan')}: {config['depth']}
    {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
    {colour_text("r", 'cyan')}: {r}
    {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f}
    {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
    {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
    {colour_text("Final Tokens", "magenta")}: {final_tokens}
    ========================\
    """
    return textwrap.dedent(message)


def export_sweep_data(data: List[Dict[str, Any]], filename):
    pd.DataFrame(data).to_csv(filename)


### Create a benchmark function (very simple)
def benchmark_compvit_milliseconds(x: torch.Tensor, model: torch.nn.Module) -> Any:
    ### Do the benchmark!
    t0 = bench.Timer(
        stmt=f"model.forward(x)",
        globals={"x": x, "model": model},
        num_threads=1,
    )

    return t0.blocked_autorange(min_run_time=8.0)


def inference(model, device, batch_size):
    ### Turn off gradient compute
    with torch.no_grad():
        ### Run Benchmark for latency, then do torch profiling!
        rand_x = torch.randn(
            size=(batch_size, 3, 224, 224), dtype=torch.float32, device=device
        )

        ### Record latency with benchmark utility
        latency_measurement = benchmark_compvit_milliseconds(rand_x, model)
        latency_mean = latency_measurement.mean * 1e3
        latency_median = latency_measurement.median * 1e3
        latency_iqr = latency_measurement.iqr * 1e3

        final_tokens = model.forward(rand_x, is_training=True)["x_norm"].shape[1]

    return latency_mean, latency_median, latency_iqr, final_tokens


def test_baseline(args):
    if args.all_dino:
        models = [
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ]
    elif args.all_deit:
        models = [
            deit_tiny_patch16_224,
            deit3_small_patch16_224,
            deit3_base_patch16_224,
            deit3_large_patch16_224,
        ]

    ### Get args, device
    device = torch.device(args.device)

    all_data = []

    # Measure dino models.
    for model_name in models:
        if args.all_dino:
            model, config = dinov2_factory(model_name=model_name)
        elif args.all_deit:
            model, config = model_name(dynamic_img_size=True)

        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr, final_tokens = inference(
            model, device, args.batch_size
        )

        message = baseline_message(
            str(model_name),
            config,
            model,
            latency_mean,
            latency_median,
            latency_iqr,
            final_tokens,
        )

        print(message)

        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
            }
        )

    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                f"{'dino' if args.all_dino else 'deit'}",
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def test_compvit(args):
    compvit_models = [
        "compvits14",
        "compvitb14",
        "compvitl14",
        "compvitg14",
    ]

    ### Get args, device
    device = torch.device(args.device)

    all_data = []
    for model_name in compvit_models:
        model, config = compvit_factory(model_name=model_name)
        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr, final_tokens = inference(
            model,
            device,
            args.batch_size,
        )

        message = compvit_message(
            model_name,
            config,
            model,
            latency_mean,
            latency_median,
            latency_iqr,
            final_tokens,
        )

        print(message)
        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "num_compressed_tokens": config["num_compressed_tokens"],
                "bottleneck_locs": config["bottleneck_loc"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
            }
        )
    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                "compvit",
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def compvit_sweep(args):
    model_name = args.sweep_model

    ### Get args, device
    device = torch.device(args.device)

    # Create iterators for the two dimensions that we can ablate over.
    # The [None] list is used when we want to fix one dimension.
    token_sweep_iter = [None]
    bottleneck_locs_iter = [None]
    if args.token_sweep:
        token_sweep = args.token_sweep  # argparse will output a list.
        token_sweep_iter = token_sweep  # the iterator is a list.
    if args.bottleneck_locs:
        start, end = args.bottleneck_locs  # argparse will output a 2 element list
        bottleneck_locs_iter = range(start, end + 1)  # the iterator is a range(...)

    all_data = []
    # Use itertools.product this takes the cartesisan product of the two iterators.
    for bottleneck_loc, compressed_tokens in itertools.product(
        bottleneck_locs_iter, token_sweep_iter
    ):
        # Control logic to handle the case when ablating over both dimensions and if one is fixed.
        if bottleneck_loc and compressed_tokens:
            model, config = compvit_factory(
                model_name=model_name,
                num_compressed_tokens=compressed_tokens,
                bottleneck_loc=bottleneck_loc,
            )
        elif bottleneck_loc:
            model, config = compvit_factory(
                model_name=model_name,
                bottleneck_loc=bottleneck_loc,
            )
        elif compressed_tokens:
            model, config = compvit_factory(
                model_name=model_name,
                num_compressed_tokens=compressed_tokens,
            )

        # Standard measurement code.
        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr, final_tokens = inference(
            model,
            device,
            args.batch_size,
        )

        message = compvit_message(
            model_name,
            config,
            model,
            latency_mean,
            latency_median,
            latency_iqr,
            final_tokens,
        )

        print(message)

        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "num_compressed_tokens": config["num_compressed_tokens"],
                "bottleneck_locs": config["bottleneck_loc"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
            }
        )

    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                model_name.replace("_", ""),
                f"bs{args.batch_size}",
                f"bnsz{config['bottleneck_size']}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def tome_sweep(args):
    model_name = args.tome_sweep_model

    ### Get args, device
    device = torch.device(args.device)
    all_data = []
    for r in range(1, 30):
        if "dinov2" in model_name:
            model, config = dinov2_tome_factory(dinov2_model_name=model_name, r=r)
        elif "deit" in model_name:
            if model_name == "deit_tiny":
                model = deit_tiny_patch16_224(dynamic_img_size=True)
                config = {"depth": 12, "embed_dim": 192}
            elif model_name == "deit_small":
                model = deit3_small_patch16_224(dynamic_img_size=True)
                config = {"depth": 12, "embed_dim": 384}
            elif model_name == "deit_base":
                model = deit3_base_patch16_224(dynamic_img_size=True)
                config = {"depth": 12, "embed_dim": 768}
            elif model_name == "deit_large":
                model = deit3_large_patch16_224(dynamic_img_size=True)
                config = {"depth": 24, "embed_dim": 1024}
            model = apply_patch(model)
            model.r = r
        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr, final_tokens = inference(
            model,
            device,
            args.batch_size,
        )

        message = tome_message(
            model_name,
            config,
            model,
            latency_mean,
            latency_median,
            latency_iqr,
            final_tokens,
            r,
        )

        print(message)
        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "r": r,
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
                "final_tokens": final_tokens,
            }
        )
    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                "tome",
                model_name.replace("_", ""),
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def topk_sweep(args):
    model_name = args.topk_sweep_model

    ### Get args, device
    device = torch.device(args.device)
    all_data = []
    for r in range(8, 48):
        model, config = dinov2_topk_factory(dinov2_model_name=model_name, r=r)
        model = model.to(device).eval()
        latency_mean, latency_median, latency_iqr, final_tokens = inference(
            model,
            device,
            args.batch_size,
        )

        message = tome_message(
            model_name,
            config,
            model,
            latency_mean,
            latency_median,
            latency_iqr,
            final_tokens,
            r,
        )

        print(message)
        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "r": r,
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
                "final_tokens": final_tokens,
            }
        )
    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                "tome",
                model_name.replace("_", ""),
                f"bs{args.batch_size}",
                f"{args.filetag}",
            ]
        )
        + ".csv"
    )
    export_sweep_data(all_data, filename)


def exit_sweep(args):
    model_name = args.exit_sweep_model
    r = args.exit_sweep_r

    ### Get args, device
    device = torch.device(args.device)
    all_data = []
    if args.exit_sweep_dino:
        model, config = dinov2_factory(model_name=model_name)
    elif args.exit_sweep_tome:
        model, config = dinov2_tome_factory(dinov2_model_name=model_name, r=r)

    model = exit_patch(model)
    model = model.to(device).eval()
    total_layers = config["depth"]
    for exit_at in range(total_layers):
        model.forward = partial(model.forward, exit_at=exit_at)
        latency_mean, latency_median, latency_iqr, final_tokens = inference(
            model,
            device,
            args.batch_size,
        )
        message = f"""\
        ========================
        {colour_text(model_name.upper(), 'green')}
        {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in model.parameters()):,}
        {colour_text("Depth", 'cyan')}: {config['depth']}
        {colour_text("Embedding Dim", 'cyan')}: {config['embed_dim']}
        {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f}
        {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
        {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
        {colour_text("Final Tokens", "magenta")}: {final_tokens}
        {colour_text("Exit At", "magenta")}: {exit_at + 1}
        ========================\
        """
        message = textwrap.dedent(message)

        print(message)
        all_data.append(
            {
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Depth": config["depth"],
                "Embedding Dim": config["embed_dim"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
                "exit_at": exit_at + 1,
            }
        )
    if args.exit_sweep_dino:
        filename = (
            "_".join(
                [
                    device_info(args).replace(" ", ""),
                    "exited",
                    model_name.replace("_", ""),
                    f"bs{args.batch_size}",
                    f"{args.filetag}",
                ]
            )
            + ".csv"
        )
    elif args.exit_sweep_tome:
        filename = (
            "_".join(
                [
                    device_info(args).replace(" ", ""),
                    "tome",
                    model_name.replace("_", ""),
                    f"r{r}" f"bs{args.batch_size}",
                    f"{args.filetag}",
                ]
            )
            + ".csv"
        )
    export_sweep_data(all_data, filename)


def test_single(args):
    ### Get args, device
    device = torch.device(args.device)

    ### Parse model name, choose appropriate factory function
    if "deit" in args.model:
        print(f"Using deit factory for {args.model}")
        model, config = compdeit_factory("tiny")
    elif "compvit" in args.model:
        print(f"Using compvit factory for {args.model}")
        model, config = compvit_factory(model_name=args.model, r=args.tome_r)
    elif "dinov2" in args.model:
        ### Wrap DinoV2 model here as necessary
        if args.wrap_tome:
            print(f"Using dinov2_tome factory for {args.model}")
            model, config = dinov2_tome_factory(
                dinov2_model_name=args.model, r=args.tome_r
            )
        else:
            print(f"Using dinov2 factory for {args.model}")
            model, config = dinov2_factory(model_name=args.model)
    else:
        raise RuntimeError(f"No factory function available for model {args.model}")

    ### Load model
    model.to(device).eval()
    print(f"# of parameters: {sum(p.numel() for p in model.parameters()):_}")

    # Inference
    latency_mean, latency_median, latency_iqr, final_tokens = inference(
        model,
        device,
        args.batch_size,
    )
    print(
        f"{args.model}| Mean/Median/IQR latency (ms) is {latency_mean:.2f} | {latency_median:.2f} | {latency_iqr:.2f}, Final Tokens: {final_tokens}"
    )


def main():
    args = parse_args()
    device_name = device_info(args)
    print(f"{colour_text('Device', 'red')}: {device_name}")

    testing_multiple = args.all_dino or args.all_compvit or args.all_deit
    if testing_multiple:
        if args.all_dino:
            print(
                f"{colour_text(f'Benchmarking DINOv2 Models @ batch size = {args.batch_size}.', 'yellow')}"
            )
            test_baseline(args)
        if args.all_compvit:
            print(
                f"{colour_text(f'Benchmarking CompViT Models  @ batch size = {args.batch_size}.', 'yellow')}"
            )
            test_compvit(args)
        if args.all_deit:
            print(
                f"{colour_text(f'Benchmarking DeiT Models @ batch size = {args.batch_size}.', 'yellow')}"
            )
            test_baseline(args)
        return 0
    elif args.compvit_sweep:
        message = f"""\
        {colour_text(f'Benchmarking CompViT Models  @ batch size = {args.batch_size}.', 'yellow')}
        {colour_text(f"Sweeping Compressed Tokens {args.token_sweep}.", 'yellow')}
        {colour_text(f"Sweeping Bottleneck Locations from {args.bottleneck_locs}.", "yellow")}\
        """
        print(textwrap.dedent(message))
        compvit_sweep(args)
        return 0
    elif args.tome_sweep:
        message = f"""\
        {colour_text(f'Benchmarking ToMe on {args.tome_sweep_model} @ batch size = {args.batch_size}.', 'yellow')}
        {colour_text(f"Sweeping r values 2 to 24.", 'yellow')}\
        """
        print(textwrap.dedent(message))
        tome_sweep(args)
        return 0
    elif args.topk_sweep:
        message = f"""\
        {colour_text(f'Benchmarking TopK on {args.topk_sweep_model} @ batch size = {args.batch_size}.', 'yellow')}
        {colour_text(f"Sweeping r values 2 to 24.", 'yellow')}\
        """
        print(textwrap.dedent(message))
        topk_sweep(args)
        return 0
    elif args.exit_sweep_dino or args.exit_sweep_tome:
        message = f"""\
        {colour_text(f'Benchmarking Models with Exiting.', 'yellow')}\
        """
        print(textwrap.dedent(message))
        exit_sweep(args)
        return 0
    else:
        test_single(args)
    return 0


if __name__ == "__main__":
    main()
