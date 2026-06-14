#!/usr/bin/env python3
"""Train TRef-SAM3 with a fixed number of random training images per epoch.

This script keeps the model, loss, validation set, and data format unchanged.
It only injects ``training.max_train_samples_per_epoch`` into a runtime copy of
the config before launching the normal SAM3 LoRA trainer.
"""

import argparse
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Train TRef-SAM3 on RefCOCO with limited random samples per epoch."
    )
    parser.add_argument(
        "--config",
        default="configs/tref_refcoco_config.yaml",
        help="Base YAML config. The file is not modified.",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=5000,
        help="Number of random training images to sample per epoch.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Single GPU id to use.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override config training.num_epochs.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output.output_dir. Defaults to '<base>_limited<N>'.",
    )
    args = parser.parse_args()

    if args.samples_per_epoch <= 0:
        raise ValueError("--samples_per_epoch must be positive")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.setdefault("training", {})
    train_cfg["max_train_samples_per_epoch"] = int(args.samples_per_epoch)
    if args.num_epochs is not None:
        train_cfg["num_epochs"] = int(args.num_epochs)

    output_cfg = config.setdefault("output", {})
    if args.output_dir:
        output_cfg["output_dir"] = args.output_dir
    else:
        base_output_dir = output_cfg.get("output_dir", "outputs/tref_refcoco")
        output_cfg["output_dir"] = f"{base_output_dir}_limited{args.samples_per_epoch}"

    runtime_dir = Path(output_cfg["output_dir"])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / "limited_runtime_config.yaml"
    with open(runtime_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print(f"Using single GPU: {args.device}")
    print(f"Runtime config: {runtime_config}")
    print(f"Training samples per epoch: {args.samples_per_epoch}")
    print(f"Output dir: {output_cfg['output_dir']}")

    from train_sam3_lora_native import SAM3TrainerNative

    trainer = SAM3TrainerNative(str(runtime_config), multi_gpu=False)
    trainer.train()


if __name__ == "__main__":
    main()
