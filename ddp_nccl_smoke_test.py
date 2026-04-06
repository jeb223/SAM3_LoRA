#!/usr/bin/env python3
"""
Minimal NCCL/DDP smoke test for multi-GPU environments.

Usage:
  python ddp_nccl_smoke_test.py --device 0 1

The launcher logic mirrors train_sam3_lora_native.py so this can be used to
separate environment/NCCL failures from model-code failures.
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.distributed as dist


def launch_distributed(args):
    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching NCCL smoke test on GPUs: {devices}", flush=True)
    print(f"Number of processes: {num_gpus}", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port",
        str(args.master_port),
        sys.argv[0],
        "--device",
        *map(str, devices),
        "--_launched_by_torchrun",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Minimal NCCL/DDP smoke test")
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU device IDs to use.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29501,
        help="Master port for distributed testing.",
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ
    if len(args.device) > 1 and not is_torchrun_subprocess:
        launch_distributed(args)
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(
        f"[Rank {rank}] init ok | local_rank={local_rank} | world_size={world_size} | device={device}",
        flush=True,
    )

    x = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] all_reduce ok | value={x.item():.1f}", flush=True)

    dist.barrier()
    print(f"[Rank {rank}] barrier ok", flush=True)

    dist.destroy_process_group()
    print(f"[Rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()
