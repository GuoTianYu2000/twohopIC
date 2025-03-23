import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import os
from omegaconf import OmegaConf
from model import *
import pickle
import numpy as np
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
# from graph_data import *


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
def load_model(date, depth, layer, head, steps, compute_loss=True, run_path=None):
    if run_path is None:
        run_path = f"{PROJECT_PATH}/runs/{date}layer{layer}head{head}"
    else:
        run_path = os.path.join(PROJECT_PATH, run_path)
    cfg = OmegaConf.load(f"{run_path}/configure.yaml")
    cfg.model_args.dim = 256
    cfg.model_args.n_heads = head
    cfg.model_args.n_layers = layer
    if getattr(cfg.data_args, "max_seq_len", None) == None:
        cfg.data_args.max_seq_len = cfg.data_args.seq_len
    ds = two_hop_format(cfg.data_args)
    cfg.model_args.vocab_size = len(ds.vocab)+len(ds.special_tokens)
    model = Transformer(cfg.model_args)
    model.cuda()
    state_path = f"{run_path}/state_{steps}.pt"
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    if compute_loss:
        seqs, seqs_ans_pos_start, seqs_ans_pos_end = next(iterate_batches(ds, num_workers=48, seed=42, batch_size=512, total_count=1))
        indices = torch.arange(cfg.data_args.max_seq_len).expand(cfg.data_args.batch_size, -1)

        x = torch.LongTensor(seqs).cuda()
        pred = model(x)
        loss = compute_loss(x, pred, seqs_ans_pos_start, seqs_ans_pos_end, indices)
        print(loss.item())
    else:
        seqs, seqs_ans_pos_start, seqs_ans_pos_end = None, None, None
    return cfg, model, seqs, seqs_ans_pos_start, seqs_ans_pos_end
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

def MakeAttnSummary(cfg, outputs_list, seqs, seqs_ans_pos_start, seqs_ans_pos_end, twoSumIndx):
    layer = cfg.model_args.n_layers
    attns = get_attns(twoSumIndx, seqs_ans_pos_start, outputs_list, layer)
    attnSummary = {}
    difLogitsSummary = {}
    for layer_idx in range(layer):
        attn_layer_reorg_mean, attn_layer_reorg = get_mean_attn(attns[layer_idx])
        logits = model.output(model.norm(model.layers[layer_idx].attention.wo(outputs_list[layer_idx]['value_states'][:, 0, :, :])))
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

