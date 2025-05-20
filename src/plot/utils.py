import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import os
from omegaconf import OmegaConf
import pickle
import numpy as np
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from model import Transformer
from two_hop_data import multi_hop_format, iterate_batches
# from graph_data import *
import matplotlib.colors as mcolors

def compute_loss(y, pred, seqs_ans_pos_start, seqs_ans_pos_end, indices, type='cross_entropy'):
    y_start = torch.LongTensor(seqs_ans_pos_start).unsqueeze(-1)
    y_end = torch.LongTensor(seqs_ans_pos_end).unsqueeze(-1)
    mask = ((indices >= y_start) & (indices < y_end)).long().cuda()
    mask_bias = -1 * ((indices < y_start) | (indices >= y_end)).long().cuda()
    masked_y = y*mask + mask_bias
    if type == 'cross_entropy':
        loss = F.cross_entropy(pred.flatten(0, 1), masked_y.flatten(0, 1), ignore_index=-1)
    elif type == '0-1':
        indiv_loss = pred[mask==1, :].argmax(-1) != masked_y[mask==1]
        indiv_loss = indiv_loss.float()
        loss = torch.mean(indiv_loss)
    return loss
def load_model(layer, head, steps, run_path, get_loss=True, device=None):
    cfg = OmegaConf.load(f"{run_path}/configure.yaml")
    if not getattr(cfg.data_args, "max_seq_len", None):
        cfg.data_args.max_seq_len = cfg.data_args.seq_len
    ds = multi_hop_format(cfg.data_args)
    cfg.model_args.vocab_size = len(ds.vocab)+len(ds.special_tokens)
    model = Transformer(cfg.model_args)
    model.to(device)
    state_path = f"{run_path}/state_{steps}.pt"
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    if get_loss:
        seqs, seqs_ans_pos_start, seqs_ans_pos_end = next(iterate_batches(ds, num_workers=48, seed=42, batch_size=512, total_count=1))
        indices = torch.arange(cfg.data_args.max_seq_len).expand(cfg.data_args.batch_size, -1)

        x = torch.LongTensor(seqs).to(device)
        y = x[:, 1:]
        x = x[:, :-1]
        pred = model(x)
        loss = compute_loss(y, pred, seqs_ans_pos_start, seqs_ans_pos_end, indices)
        print(loss.item())
    else:
        seqs, seqs_ans_pos_start, seqs_ans_pos_end = None, None, None
    return cfg, model, seqs, seqs_ans_pos_start, seqs_ans_pos_end
def plot_attns(seqs, outputs_list, seq_indices, seq_start, seq_len, layer, head, numToChr=None, save_dir="neurips_figures"):
    if not isinstance(seq_indices, int):
        raise ValueError("We only support one sequence for now")
    
    plot_count = 0
    for layer_idx in range(layer):
        for head_idx in range(head):
            attns = outputs_list[layer_idx]['attn_logits'].detach()
            attns_plot = attns[seq_indices, head_idx, seq_start:seq_len, seq_start:seq_len].cpu().numpy()
            
            print(f"Layer {layer_idx}, Head {head_idx}")
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Create labels
            nums = seqs[seq_indices][seq_start:seq_len]
            if numToChr is not None:
                labels = [numToChr.get(num, '') for num in nums]
                labels[0] = r'$\langle s \rangle$'
            else:
                labels = nums
            
            # Create mask with the same shape as attns_plot
            mask = np.triu(np.ones_like(attns_plot), k=1)

            # Determine whether to show color bar
            show_cbar = plot_count >= 2
            gamma = 1

            # Plot with seaborn's default colormap
            sns.heatmap(
                attns_plot, mask=mask,
                xticklabels=labels, yticklabels=labels,
                cbar=show_cbar,
                cmap="Blues", norm=mcolors.PowerNorm(gamma=gamma), 
                cbar_kws={
                    "shrink": 1,     # Keep original height
                    "pad": 0.02,       # Distance from plot
                    "fraction": 0.03,  # Width of the colorbar (smaller = thinner)
                    "aspect": 50,      # More slender aspect ratio
                } if show_cbar else {}
            )

            if show_cbar:
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=30)  # Set tick label size here
            
            # Increase font size for tick labels
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            
            # Add rectangle outlines with thinner gray lines for better visibility
            for i in range(len(attns_plot)):
                for j in range(i+1):
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=0.5, alpha=0.7))
                
            ax.set_ylim(len(attns_plot), 0)
            ax.set_xlim(0, len(attns_plot))
            
            # Rotate x-tick labels by 90 degrees
            plt.xticks(rotation=90)
            
            # Adjust layout for better appearance
            plt.tight_layout()
            
            # Save the figure if save_dir is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"layer_{layer_idx}_head_{head_idx}.pdf"), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            plot_count += 1
def getSumIndx(seqs, twoSum, ):
    getIndx = lambda x, t, i: torch.nonzero(torch.isin(x, torch.tensor([t]))).squeeze(-1)[i].item()
    twoSumIndx = []
    for seq, twohop in zip(torch.tensor(seqs), twoSum):
        twoSumIndxtmp = {'target': {'start': [], 'mid1': [], 'mid2': [], 'end': []}, 'noise': {'start': [], 'mid1': [], 'mid2': [], 'end': []}}
        for k1, group in twohop.items():
            for s, m, e in zip(group['start'], group['mid'], group['end']):
                twoSumIndxtmp[k1]['start'].append(getIndx(seq, s, 0))
                twoSumIndxtmp[k1]['mid1'].append(getIndx(seq, m, 0))
                twoSumIndxtmp[k1]['mid2'].append(getIndx(seq, m, 1))
                twoSumIndxtmp[k1]['end'].append(getIndx(seq, e, 0))
        twoSumIndx.append(twoSumIndxtmp)
    return twoSumIndx
def get_mean_attn(attn_layer):
    attn_layer_reorg = {}
    for attn_layer_seq in attn_layer.values():
        for k, v in attn_layer_seq.items():
            attn_layer_reorg.setdefault(k, {})
            for pair, pair_attn in v.items():
                attn_layer_reorg[k].setdefault(pair, []).append(np.mean(pair_attn).item())
    attn_layer_reorg_mean = {}
    for k, v in attn_layer_reorg.items():
        attn_layer_reorg_mean[k] = {}
        for pair, pair_attn in v.items():
            attn_layer_reorg_mean[k][pair] = np.mean(pair_attn).item()
    return attn_layer_reorg_mean, attn_layer_reorg

def get_attns(twoSumIndx, seqs_ans_pos_start, outputs_list, layer):
    def pairwise_attns(attns, s, m1, m2, e, qi, outputs, seqi, layer_idx):
        tmpIndx = [('start', s), ('mid1', m1), ('mid2', m2), ('end', e), ('query', qi)]
        for i in range(len(tmpIndx)):
            for j in range(i+1, len(tmpIndx)):
                namej, namei = tmpIndx[j][0], tmpIndx[i][0]
                idxj, idxi = tmpIndx[j][1], tmpIndx[i][1]
                attns.setdefault((namej, namei), []).append(outputs['attn_weights'][seqi, 0, idxj, idxi].item())
        attns[('c', 'p')] = [outputs['attn_weights'][seqi, 0, idx, idx-1].item() for idx in range(2, qi, 2)]
        return
    attns = {}
    for layer_idx in range(layer):
        attn_layer = {}
        for seqi, (qi, twohopIndx) in enumerate(zip(seqs_ans_pos_start, twoSumIndx)):
            attn_layer_seq = {}
            for k, group in twohopIndx.items():
                attn_layer_seq[k] = {}
                for s, m1, m2, e in zip(group['start'], group['mid1'], group['mid2'], group['end']):
                    pairwise_attns(attn_layer_seq[k], s, m1, m2, e, qi, outputs_list[layer_idx], seqi, layer_idx)
            attn_layer[seqi] = attn_layer_seq
        attns[layer_idx] = attn_layer
    return attns

def MakeAttnSummary(cfg, outputs_list, seqs, seqs_ans_pos_start, seqs_ans_pos_end, twoSumIndx, used_model):
    layer = cfg.model_args.n_layers
    attns = get_attns(twoSumIndx, seqs_ans_pos_start, outputs_list, layer)
    attnSummary = {}
    difLogitsSummary = {}
    for layer_idx in range(layer):
        attn_layer_reorg_mean, attn_layer_reorg = get_mean_attn(attns[layer_idx])
        logits = used_model.output(used_model.norm(used_model.layers[layer_idx].attention.wo(outputs_list[layer_idx]['value_states'][:, 0, :, :])))
        logits = logits[: , :seqs_ans_pos_start[0]+1, :]
        x = torch.LongTensor(seqs).to(logits.device)[:, :seqs_ans_pos_start[0]+1].unsqueeze(-1)
        target_logits = torch.gather(logits, 2, x)
        other_logits = (logits.sum(-1).unsqueeze(-1) - target_logits) / (logits.shape[-1] - 1)
        dif_logits = (target_logits - other_logits).squeeze(-1)
        dif_logits_dict = {}
        for tok in torch.unique(x):
            dif_logits_dict[tok.item()] = dif_logits[x.squeeze(-1) == tok].mean().item()
        attnSummary[layer_idx] = attn_layer_reorg_mean
        difLogitsSummary[layer_idx] = dif_logits_dict
    return attnSummary, difLogitsSummary

