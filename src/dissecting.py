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
import seaborn as sns
import matplotlib.pyplot as plt
from .plot.utils import *
from .plot.plot_utils import *
from tqdm import tqdm

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
import os

"""values that you need to change or provide in the command line"""
DEVICE = "cuda:3"
PROJECT_PATH = "/home/tianyu/TwoHopIC" # the path of your working directory
date = "2025-01-13" # the date of the model
layer = 3 # the layer of the model
head = 1 # the head of the model
hopk = 2 # the hopk of the model
steps = 10000 # the steps of the model  
run_path = f"pre-icml/L{layer}_hopk{hopk}" # the run path of the model


from .model import *
from .two_hop_data import *


device = DEVICE
torch.cuda.set_device(device)



@dataclass
class DynamicArgs:
    project_path: str = PROJECT_PATH
    date: str = date
    layer: int = layer
    head: int = head
    hopk: int = hopk
    steps: int = steps
    device_num: int = int(DEVICE.split(":")[-1])
    run_path: Optional[str] = run_path
    test: bool = False


if __name__ == "__main__":
    args = DynamicArgs()
    args = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(device)
    model_date, layer, head, hopk, steps, run_path, project_path = args.date, args.layer, args.head, args.hopk, args.steps, args.run_path, args.project_path
    # run_path=f"pre-icml/L{layer}_hopk{hopk}"
    cfg, model, seqs, seqs_ans_pos_start, seqs_ans_pos_end = load_model(model_date, None, layer, 1, steps, compute_loss=False, run_path=run_path, project_path=project_path, device=device)
    if cfg.task_name == "twoHop":
        ds = two_hop_format(cfg.data_args)
    elif cfg.task_name == "multiHop":
        ds = multi_hop_format(cfg.data_args)
    seqs, seqs_ans_pos_start, seqs_ans_pos_end, twoSum, oneSum = next(iterate_batches(ds, num_workers=48, seed=42, batch_size=512, total_count=1, withhops=True))
    twoSumIndx = getSumIndx(seqs, twoSum, )

    if args.test:
        cfg.fine_grid_log = 0
    if cfg.fine_grid_log == 0:
        log_steps = [0, cfg.max_iters]
    else:
        log_steps = np.arange(0, cfg.fine_grid_log, 5).tolist()
        log_steps.extend(np.arange(cfg.fine_grid_log, cfg.max_iters, 20).tolist())
        log_steps = [int(x) for x in log_steps]
    attnSummaryStep = {}
    difLogitsSummaryStep = {}
    
    for steps in tqdm(log_steps):
        cfg, model, _, _, _ = load_model(model_date, None, layer, 1, steps, compute_loss=False, run_path=run_path, project_path=project_path, device=device)
        hook = forward_hook([], '')
        pred, outputs_list = model.modified_forward_with_hook(torch.LongTensor(seqs)[:, :-1].cuda(), hook)

        
        attnSummary, difLogitsSummary = MakeAttnSummary(cfg, outputs_list, seqs, seqs_ans_pos_start, seqs_ans_pos_end, twoSumIndx, model)
        attnSummaryStep[steps] = attnSummary
        difLogitsSummaryStep[steps] = difLogitsSummary
    
    save_dir = os.path.join(project_path, run_path)
    os.makedirs(os.path.join(save_dir, "dynamics"), exist_ok=True)

    attnSummaryStep_serializable = {k: {k1: {k2: {k3[0]+'->'+k3[1]: v3 for k3, v3 in v2.items()} for k2, v2 in v1.items()} for k1, v1 in v.items()} for k, v in attnSummaryStep.items()}
    with open(os.path.join(save_dir, f"dynamics/attnSummaryStep_serializable.json"), "w") as f:
        json.dump(attnSummaryStep_serializable, f)
    with open(os.path.join(save_dir, f"dynamics/difLogitsSummaryStep.json"), "w") as f:
        json.dump(difLogitsSummaryStep, f)


