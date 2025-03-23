from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import json
import pickle
import time
import torch
import sys
import yaml
import os
import pdb
import wandb

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from two_hop_data import *
from model import *
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

"""Args that you need to change"""
@dataclass
class WandbArgs:
    project: str = 'twoHop'
    entity: str = 'tianyu_guo'
    name: str = 'rerun'

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) # RZ: Sets the seed for generating random numbers for all devices (both CPU and CUDA).
    torch.manual_seed(seed) # RZ: Specifically sets the seed for all CUDA GPUs for generating random numbers. This is necessary for ensuring that all GPUs have the same initial random state.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # RZ: Some operations in CUDA are non-deterministic by default. To ensure determinism, you might need to set additional flags in PyTorch. However, this could potentially impact performance.
    torch.backends.cudnn.benchmark = False


@dataclass
class OptimArgs:
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD
    batch_size: int = 64
    use_sgd: bool = False  # otherwise use AdamW
    seperate_lr: bool = False
    massiv_lr: float = 0.0003
    copy_lr: float = 0.03



@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    wandb_args: WandbArgs
    use_simple_model: Optional[bool] = False
    max_iters: int=10000
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    num_data_workers: int = 48
    save_dir: Optional[str] = None
    fine_grid_log: int = 0
    root_dir: str = ''
    task_name: str = ''
    seperate_loss: bool = False
    device_num: int = 2
    seed: int = 42


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs(),
           wandb_args=WandbArgs(),
        )
    cli_args = OmegaConf.from_cli()
    cfg = OmegaConf.merge(OmegaConf.structured(args), cli_args)
    # cfg.model_args.bos_num = cfg.data_args.bos_num
    set_random_seed(cfg.seed)
    torch.cuda.set_device(cfg.device_num)
    if cfg.task_name == "twoHop":
        ds = two_hop_format(cfg.data_args)
    elif cfg.task_name == "multiHop":
        ds = multi_hop_format(cfg.data_args)

    # Save the data
    if cfg.save_dir is None:
        if cfg.model_args.dim == 256 and cfg.optim_args.learning_rate == 0.0003:
            save_dir = os.path.join("runs", str(date.today())+"layer"+str(cfg.model_args.n_layers)+"head"+str(cfg.model_args.n_heads))
        else:
            assert cfg.model_args.n_layers == 3
            save_dir = os.path.join("runs", str(date.today())+"L3"+"dim"+str(cfg.model_args.dim)+"lr"+str(cfg.optim_args.learning_rate))
    else:
        save_dir = os.path.join("runs", cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cfg.save_dir = save_dir
    wandb.init(
            dir=str(save_dir),
            project=cfg.wandb_args.project,
            entity=cfg.wandb_args.entity,
            name=str(save_dir),
        )
    cfg.model_args.vocab_size = len(ds.vocab)+len(ds.special_tokens)
    OmegaConf.save(cfg, os.path.join(save_dir, "configure.yaml"))
    model = Transformer(cfg.model_args)
    model.cuda()

    # optim
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim_args.learning_rate,
        weight_decay=cfg.optim_args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8)
    if cfg.fine_grid_log == 0:
        log_steps = np.linspace(0, cfg.max_iters, 6).tolist()
        log_steps = [int(x) for x in log_steps]
    else:
        log_steps = np.arange(0, cfg.fine_grid_log, 5).tolist()
        log_steps.extend(np.arange(cfg.fine_grid_log, cfg.max_iters, 20).tolist())
        log_steps = [int(x) for x in log_steps]
    log_steps = set(log_steps)
    indices = torch.arange(cfg.data_args.seq_len).expand(cfg.data_args.batch_size, -1)

    pbar = tqdm(enumerate(iterate_batches(ds, num_workers=cfg.num_data_workers, seed=cfg.seed, batch_size=cfg.data_args.batch_size, total_count=cfg.max_iters)), total=cfg.max_iters)

    for i, (seqs, seqs_ans_pos_start, seqs_ans_pos_end) in pbar:
        if i in log_steps:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i}.pt"))
        x = torch.LongTensor(seqs).cuda()
        y = x[:, 1:]
        x = x[:, :-1]
        optimizer.zero_grad()
        pred = model(x)

        loss = compute_loss(y, pred, seqs_ans_pos_start, seqs_ans_pos_end, indices)
        loss.backward()

        wandb.log({"loss": loss.item(), "step": i})
        pbar.set_description(f"loss: {loss.item()}")

        optimizer.step()

    # save the last state
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": i+1,
    }
    torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i+1}.pt"))
