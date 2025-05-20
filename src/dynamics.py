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
# os.chdir("/data/your_name_guo/birth")
from .model import *
from .two_hop_data import *


"""values that you need to change or provide in the command line"""
DEVICE = "cuda:3"
PROJECT_PATH = "/home/your_name/TwoHopIC" # the path of your working directory
date = "2025-01-13" # the date of the model
layer = 3 # the layer of the model
head = 1 # the head of the model
hopk = 2 # the hopk of the model
steps = 10000 # the steps of the model

def compute_loss(y, pred, seqs_ans_pos_start, seqs_ans_pos_end, indices, type='cross_entropy'):
    y_start = torch.LongTensor(seqs_ans_pos_start).unsqueeze(-1)
    y_end = torch.LongTensor(seqs_ans_pos_end).unsqueeze(-1)
    # mask_pred = (indices >= y_pos).long().cuda()
    mask = ((indices >= y_start) & (indices < y_end)).long().cuda()
    mask_bias = -1 * ((indices < y_start) | (indices >= y_end)).long().cuda()
    # masked_pred = pred * mask_pred.unsqueeze(-1)
    # masked_x = x*mask
    masked_y = y*mask + mask_bias
    # loss = F.cross_entropy(masked_pred[:, :-1, :].flatten(0, 1), masked_x[:, 1:].flatten(0, 1), reduction='none')
    if type == 'cross_entropy':
        loss = F.cross_entropy(pred.flatten(0, 1), masked_y.flatten(0, 1), ignore_index=-1)
    elif type == '0-1':
        indiv_loss = pred[mask==1, :].argmax(-1) != masked_y[mask==1]
        indiv_loss = indiv_loss.float()
        loss = torch.mean(indiv_loss)
    return loss


def plot_attns(outputs_list, seq_idx, seq_start, seq_len):
    for layer_idx in range(layer):
        for head_idx in range(head):
            attns = outputs_list[layer_idx]['attn_weights'].detach().cpu().numpy()
            attns_plot = attns[seq_idx, head_idx, seq_start:seq_len, seq_start:seq_len]
            mask = 1 - np.tril(np.ones_like(attns_plot))
            # label_text = text_test
            print(f"Layer {layer_idx}, Head {head_idx}")
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                attns_plot, mask=mask,
                cmap="Blues", xticklabels=seqs[seq_idx][seq_start:seq_len], yticklabels=[],
                vmin=0, vmax=1, cbar=False, cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect":50, "ticks": [0, 1]}
            )
            plt.show()

def get_seperate_prob(model, seqs, seqs_ans_pos_start, seqs_ans_pos_end, twoSum, oneSum):
    hook = forward_hook([], '')
    pred, outputs_list = model.modified_forward_with_hook(torch.LongTensor(seqs)[:, :-1].cuda(), hook)
    predProb = F.softmax(pred, -1)[range(pred.shape[0]), seqs_ans_pos_start, :]
    twoProbSum, oneProbSum = {'target': {'start': [], 'mid': [], 'end': []}, 'noise': {'start': [], 'mid': [], 'end': []}}, {'target': {'start': [], 'end': []}, 'noise': {'start': [], 'end': []}}
    contextSum = []
    endSum = []
    for i, prob in enumerate(predProb):
        contextSumi = 0
        endSumi = 0
        for k in twoProbSum.keys():
            for k1 in twoProbSum[k].keys():
                if twoSum[i][k][k1] == []:
                    continue
                probSum = prob[twoSum[i][k][k1]].sum().item()
                twoProbSum[k][k1].append(probSum)
                contextSumi += probSum
                if k1 != 'start':
                    endSumi += probSum
        for k in oneProbSum.keys():
            for k1 in oneProbSum[k].keys():
                if oneSum[i][k][k1] == []:
                    continue
                probSum = prob[oneSum[i][k][k1]].sum().item()
                oneProbSum[k][k1].append(probSum)
                contextSumi += probSum
                if k1 != 'start':
                    endSumi += probSum
        contextSum.append(contextSumi)
        endSum.append(endSumi)
    for k in oneProbSum.keys():
        for k1 in oneProbSum[k].keys():
            if oneProbSum[k][k1]:
                oneProbSum[k][k1] = np.mean(oneProbSum[k][k1]).item()
            else:
                oneProbSum[k][k1] = None
    for k in twoProbSum.keys():
        for k1 in twoProbSum[k].keys():
            if twoProbSum[k][k1]:
                twoProbSum[k][k1] = np.mean(twoProbSum[k][k1]).item()
            else:
                twoProbSum[k][k1] = None
    contextSum = np.mean(contextSum).item()
    endSum = np.mean(endSum).item()
    return twoProbSum, oneProbSum, contextSum, endSum

@dataclass
class DynamicArgs:
    project_path: str = PROJECT_PATH
    date: str="2025-01-13"
    layer: int=3
    head: int=1
    hopk: int=2
    steps: int=10000
    device_num: int = 2
    run_path: Optional[str] = None
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

    if args.test:
        cfg.fine_grid_log = 0
    if cfg.fine_grid_log == 0:
        log_steps = np.linspace(0, cfg.max_iters, 6).tolist()
        log_steps = [int(x) for x in log_steps]
    else:
        log_steps = np.arange(0, cfg.fine_grid_log, 5).tolist()
        log_steps.extend(np.arange(cfg.fine_grid_log, cfg.max_iters, 20).tolist())
        log_steps = [int(x) for x in log_steps]
    twoProbSumDict = {}
    oneProbSumDict = {}
    contextSumDict = {}
    endSumDict = {}
    for steps in tqdm(log_steps):
        cfg, model, _, _, _ = load_model(model_date, None, layer, 1, steps, compute_loss=False, run_path=run_path, project_path=project_path, device=device)
        import pdb; pdb.set_trace()
        twoProbSum, oneProbSum, contextSum, endSum = get_seperate_prob(model, seqs, seqs_ans_pos_start, seqs_ans_pos_end, twoSum, oneSum)
        twoProbSumDict[steps] = twoProbSum
        oneProbSumDict[steps] = oneProbSum
        contextSumDict[steps] = contextSum
        endSumDict[steps] = endSum
    
    save_dir = os.path.join(project_path, run_path)
    os.makedirs(os.path.join(save_dir, "dynamics"), exist_ok=True)
    with open(os.path.join(save_dir, f"dynamics/twoProbSumDict.json"), "w") as f:
        json.dump(twoProbSumDict, f)
    with open(os.path.join(save_dir, f"dynamics/oneProbSumDict.json"), "w") as f:
        json.dump(oneProbSumDict, f)
    with open(os.path.join(save_dir, f"dynamics/contextSumDict.json"), "w") as f:
        json.dump(contextSumDict, f)
    with open(os.path.join(save_dir, f"dynamics/endSumDict.json"), "w") as f:
        json.dump(endSumDict, f)
