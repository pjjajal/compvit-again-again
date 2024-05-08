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
from compvit.layers.compressor import Compressor
from benchmark import (
    colour_text,
    device_info,
    benchmark_compvit_milliseconds,
    baseline_message,
)


from dataclasses import dataclass

COMPRESSOR_SIZE = {
    "small": {
        "dim": 384,
        "num_heads": 6,
    },
    "base": {
        "dim": 768,
        "num_heads": 12,
    },
    "large": {
        "dim": 1024,
        "num_heads": 16,
    },
    "huge": {
        "dim": 1536,
        "num_heads": 24,
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Benchmarking Code")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--size", choices=["small", "base", "large", "huge"])
    parser.add_argument("--num_compressed_tokens", type=int, nargs="+")
    parser.add_argument("--input_sizes", type=int, nargs="+")
    return parser.parse_args()


def export_sweep_data(data: List[Dict[str, Any]], filename):
    pd.DataFrame(data).to_csv(filename)


def inference(model, device, batch_size, input_size, dim):
    ### Turn off gradient compute
    with torch.no_grad():
        ### Run Benchmark for latency, then do torch profiling!
        rand_x = torch.randn(
            size=(batch_size, input_size, dim), dtype=torch.float32, device=device
        )

        ### Record latency with benchmark utility
        latency_measurement = benchmark_compvit_milliseconds(rand_x, model)
        latency_mean = latency_measurement.mean * 1e3
        latency_median = latency_measurement.median * 1e3
        latency_iqr = latency_measurement.iqr * 1e3

    return latency_mean, latency_median, latency_iqr


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Pairs num_compressed_token and input size pairs to ablate over.
    pairs = filter(
        lambda x: x[0] <= x[1],
        itertools.product(args.num_compressed_tokens, args.input_sizes),
    )

    data = []
    for num_compressed_tokens, input_size in pairs:
        compressor = Compressor(
            mlp_ratio=4,
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
            init_values=1.0,
            num_compressed_tokens=num_compressed_tokens,
            **COMPRESSOR_SIZE[args.size],
        ).to(device)
        compressor.eval()

        latency_mean, latency_median, latency_iqr = inference(
            compressor,
            device,
            args.batch_size,
            input_size,
            COMPRESSOR_SIZE[args.size]["dim"],
        )

        message = f"""\
            ========================
            {colour_text(f"Compressor {args.size}".upper(), 'green')}
            {colour_text("Parameters", 'cyan')}: {sum(p.numel() for p in compressor.parameters()):,}
            {colour_text("Dim", 'cyan')}: {COMPRESSOR_SIZE[args.size]['dim']}
            {colour_text("Num Heads", 'cyan')}: {COMPRESSOR_SIZE[args.size]['num_heads']}
            {colour_text("Mean (ms)", "magenta")}: {latency_mean:.2f} 
            {colour_text("Median (ms)", "magenta")}: {latency_median:.2f}
            {colour_text("IQR (ms)", "magenta")}: {latency_iqr:.2f}
            {colour_text("Input Tokens", "magenta")}: {input_size}
            {colour_text("Compressed Tokens", "magenta")}: {num_compressed_tokens}
            ========================\
            """
        message = textwrap.dedent(message)
        print(message)

        data.append(
            {
                "Parameters": sum(p.numel() for p in compressor.parameters()),
                "Embedding Dim": COMPRESSOR_SIZE[args.size]["dim"],
                "Num Heads": COMPRESSOR_SIZE[args.size]["num_heads"],
                "Mean (ms)": latency_mean,
                "Median (ms)": latency_median,
                "IQR (ms)": latency_iqr,
                "Input Tokens": input_size,
                "Compressed Tokens": num_compressed_tokens,
            }
        )

    filename = f"compressor_ablate_{args.size}.csv"
    export_sweep_data(data, filename)


if __name__ == "__main__":
    main()
