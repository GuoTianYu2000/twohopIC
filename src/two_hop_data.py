import networkx as nx
import random
from dataclasses import dataclass, field
import numpy as np
import json
import os
from omegaconf import OmegaConf
from datetime import date
import sys
import multiprocessing as mp
import pickle
from tqdm import tqdm
import torch
from torch.nn import functional as F
import pdb
from typing import List, Optional, Tuple

@dataclass
class DataArgs:
    nodes_vocab_size: int=30
    graph_size: tuple[int]=tuple(range(15, 26))
    edge_probability: float=0.2
    twohops: int=4
    special_tokens: dict[str, int]=field(default_factory=lambda: {"<BOS>": 0, "<EDGE>": 1, "<ANS>": 2, "<EOS>": 3})
    seq_len: int=24
    use_bos: bool=False
    batch_size: int=512
    clean: bool=True
    hopk: Optional[int]=2

class two_hop_format:
    def __init__(self, args, **kwargs) -> None:
        assert args.seq_len % 2 == 0
        self.nodeStart = max(args.special_tokens.values()) + 1
        self.vocab = range(self.nodeStart, args.nodes_vocab_size+self.nodeStart)
        self.special_tokens = args.special_tokens
        self.seq_len = args.seq_len
        self.twohops = args.twohops
        self.max_edge = (self.seq_len-1) // 2 if args.use_bos else (self.seq_len-2) // 2
        self.use_bos = args.use_bos
        self.special_config = {}

    def gen_hops(self, rng, ):
        twohopsNum = rng.choice(range(1, self.twohops+1))
        onehopsNum = self.max_edge - 2*twohopsNum
        nodesNum = twohopsNum*3 + onehopsNum*2
        nodes = rng.choice(self.vocab, size=nodesNum, replace=False)
        twohopNodes, onehopNodes = nodes[:twohopsNum*3], nodes[twohopsNum*3:]


        twohops = [((twohopNodes[i], twohopNodes[i+1]), (twohopNodes[i+1], twohopNodes[i+2])) for i in range(0, twohopsNum*3, 3)]
        onehops = [(onehopNodes[i], onehopNodes[i+1]) for i in range(0, onehopsNum*2, 2)]

        return twohops, onehops
    
    def gen_seq(self, rng, withhops=False):
        twohops, onehops = self.gen_hops(rng)
        seq = self.hops_to_seq(twohops, onehops, rng)
        ans = rng.choice(twohops)
        seq += [ans[0][0], ans[-1][-1]]
        ans_start = len(seq) - 2
        ans_end = len(seq) - 1
        seq += [self.special_tokens["<EOS>"]] * (self.seq_len + 1 - len(seq))
        if withhops:
            twoSum, oneSum = self.summarise_hops(twohops, onehops, ans)
            return seq, ans_start, ans_end, twoSum, oneSum
        else:
            return seq, ans_start, ans_end

    def hops_to_seq(self, twohops, onehops, rng):
        onehops = onehops[:]
        for h in twohops:
            p1 = rng.integers(0, len(onehops)+1)
            p2 = rng.integers(p1+1, len(onehops)+2)
            onehops.insert(p1, h[0])
            onehops.insert(p2, h[1])
        seq = [item for sublist in onehops for item in sublist]
        if self.use_bos:
            seq = [self.special_tokens["<BOS>"]] + seq
        return seq

    def summarise_hops(self, twohops, onehops, ans):
        twoSum, oneSum = {'target': {'start': [], 'mid': [], 'end': []}, 'noise': {'start': [], 'mid': [], 'end': []}}, {'target': {'start': [], 'end': []}, 'noise': {'start': [], 'end': []}}
        query = ans[0][0]
        twoSum['noise']['start'] = [h[0][0] for h in twohops if h[0][0] != query]
        twoSum['noise']['mid'] = [h[0][1] for h in twohops if h[0][0] != query]
        twoSum['noise']['end'] = [h[1][1] for h in twohops if h[0][0] != query]
        twoSum['target']['start'] = [ans[0][0]]
        twoSum['target']['mid'] = [ans[0][1]]
        twoSum['target']['end'] = [ans[1][1]]
        oneSum['noise']['start'] = [h[0] for h in onehops]
        oneSum['noise']['end'] = [h[1] for h in onehops]
        return twoSum, oneSum



    # def seqs_to_hops(self, seqs, seqs_ans_pos_start, seqs_ans_pos_end):
    #     twohops, onehops = [], []
    #     starti = 1 if self.use_bos else 0
    #     for k, (seq, ans_pos_start, ans_pos_end) in enumerate(zip(seqs, seqs_ans_pos_start, seqs_ans_pos_end)):
    #         twohop, onehop = {'target': {'start': [], 'mid': [], 'end': []}, 'noise': {'start': [], 'mid': [], 'end': []}}, {'target': {'start': [], 'end': []}, 'noise': {'start': [], 'end': []}}
    #         q, a = seq[ans_pos_start], seq[ans_pos_end]
    #         for i in range(starti, ans_pos_start, 2):
    #             start, startidx, end, endidx = seq[i], i, seq[i+1], i+1
    #             ends.append([k, endidx, end])
    #             seqEnds.add(end)
    #             contexts.append([k, startidx, start])
    #             contexts.append([k, endidx, end])
    #             if start in seqEnds:
    #                 twohopEnds.append([k, endidx, end])
    #                 if end != seq[ans_pos_end]:
    #                     noiseTwohopEnds.append([k, endidx, end])
    #                     foundnoise = True
    #             else:
    #                 onehopEnds.append([k, endidx, end])
    #         if foundnoise:
    #             twohopNoiseNum += 1
    #     return twohopEnds, noiseTwohopEnds, onehopEnds, ends, contexts, twohopNoiseNum

class multi_hop_format:
    def __init__(self, args, **kwargs) -> None:
        self.nodeStart = max(args.special_tokens.values()) + 1
        self.vocab = range(self.nodeStart, args.nodes_vocab_size+self.nodeStart)
        self.special_tokens = args.special_tokens
        self.seq_len = args.seq_len
        self.hopk = args.hopk
        self.max_edge = (self.seq_len-2) // 2
        self.hopsNum = self.max_edge // self.hopk
        self.use_bos = args.use_bos
        self.special_config = {}

    def gen_hopk(self, i, nodes):
        return [(nodes[j], nodes[j+1]) for j in range(i, i+self.hopk)]

    def gen_hops(self, rng, ):
        nodesNum = self.hopsNum * (self.hopk + 1)
        nodes = rng.choice(self.vocab, size=nodesNum, replace=False)
        hops = [self.gen_hopk(i, nodes) for i in range(len(nodes)) if i % (self.hopk+1) == 0]
        return hops
    
    def gen_seq(self, rng, withhops=False):
        hops = self.gen_hops(rng)
        seq = self.hops_to_seq(hops, rng)
        ans = rng.choice(hops)
        seq += [ans[0][0], ans[-1][-1]]
        ans_start = len(seq) - 2
        ans_end = len(seq) - 1
        seq += [self.special_tokens["<EOS>"]] * (self.seq_len + 1 - len(seq))
        if withhops:
            assert self.hopk == 2
            twoSum, oneSum = self.summarise_hops(hops, [], ans)
            return seq, ans_start, ans_end, twoSum, oneSum
        else:
            return seq, ans_start, ans_end

    def hops_to_seq(self, hops, rng):
        hoptmp = []
        weights = np.array([len(h) for h in hops])
        wsum = weights.sum()
        while wsum > 0:
            hidx = rng.choice(range(len(hops)), p=weights/wsum)
            hoptmp.append(hops[hidx][len(hops[hidx])-weights[hidx]])
            weights[hidx] -= 1
            wsum = weights.sum()
        seq = [item for sublist in hoptmp for item in sublist]
        if self.use_bos:
            seq = [self.special_tokens["<BOS>"]] + seq
        return seq
    
    def summarise_hops(self, twohops, onehops, ans):
        twoSum, oneSum = {'target': {'start': [], 'mid': [], 'end': []}, 'noise': {'start': [], 'mid': [], 'end': []}}, {'target': {'start': [], 'end': []}, 'noise': {'start': [], 'end': []}}
        query = ans[0][0]
        twoSum['noise']['start'] = [h[0][0] for h in twohops if h[0][0] != query]
        twoSum['noise']['mid'] = [h[0][1] for h in twohops if h[0][0] != query]
        twoSum['noise']['end'] = [h[1][1] for h in twohops if h[0][0] != query]
        twoSum['target']['start'] = [ans[0][0]]
        twoSum['target']['mid'] = [ans[0][1]]
        twoSum['target']['end'] = [ans[1][1]]
        oneSum['noise']['start'] = [h[0] for h in onehops]
        oneSum['noise']['end'] = [h[1] for h in onehops]
        return twoSum, oneSum

def iterate_batches(ds: two_hop_format,
                    num_workers: int = 60,
                    seed: int = 42,
                    batch_size: int = 256,
                    total_count: int = 10000,
                    target: str = '',
                    **kwargs):
    stop_event = mp.Event()
    def worker(queue, rng, stop_event, target, ):
        while not stop_event.is_set():
            try:
                if target:
                    outputs = ds.gen_seq_for_probe(rng, target, **kwargs)
                else:
                    outputs = ds.gen_seq(rng, **kwargs)
                queue.put(outputs)
            except:
                print(f"Worker encountered an error")
    
    q = mp.Queue(maxsize=int(1000))
    processes = [mp.Process(target=worker, args=(q, np.random.default_rng([seed, i]), stop_event, target, )) for i in range(num_workers)]
    for p in processes:
        p.start()
    
    try:
        for _ in range(total_count):
            outputs = q.get()
            outputs_list = [[o] for o in outputs]
            for i in range(1, batch_size):
                outputs = q.get()
                for i in range(len(outputs_list)):
                    outputs_list[i].append(outputs[i])
            yield outputs_list
    finally:
        for p in processes:
            p.terminate()
            p.join()
        q.close()
        q.join_thread()

def compute_loss(y, pred, seqs_ans_pos_start, seqs_ans_pos_end, indices):
    y_start = torch.LongTensor(seqs_ans_pos_start).unsqueeze(-1)
    y_end = torch.LongTensor(seqs_ans_pos_end).unsqueeze(-1)

    mask = ((indices >= y_start) & (indices < y_end)).long().cuda()
    mask_bias = -1 * ((indices < y_start) | (indices >= y_end)).long().cuda()


    masked_y = y*mask + mask_bias


    loss = F.cross_entropy(pred.flatten(0, 1), masked_y.flatten(0, 1), ignore_index=-1)
    return loss

# Example Usage
if __name__ == "__main__":
    # Generate a DAG
    args = DataArgs()
    args = OmegaConf.structured(args)
    """twohop test"""
    # ds = two_hop_format(args)
    """"multi-hop test"""
    args.use_bos = True
    args.hopk = 2
    args.nodes_vocab_size = 48
    args.seq_len = 64
    ds = multi_hop_format(args)

    seqs, seqs_ans_pos_start, seqs_ans_pos_end = [], [], []
    rng = np.random.default_rng(0)
    print(ds.gen_seq(rng, withhops=True))
    total_count = int(10)
    for i, (seqs, seqs_ans_pos_start, seqs_ans_pos_end) in tqdm(enumerate(iterate_batches(ds, num_workers=48, seed=42, batch_size=args.batch_size, total_count=total_count)), total=total_count):
        pass
    seqs = np.array(seqs, dtype=int)
    seqs_ans_pos_start = np.array(seqs_ans_pos_start, dtype=int)
    seqs_ans_pos_end = np.array(seqs_ans_pos_end, dtype=int)
    print(seqs[0, :])
    print(seqs_ans_pos_start)
    compute_loss(torch.LongTensor(seqs[:, 1:]).cuda(), None, seqs_ans_pos_start, seqs_ans_pos_end, torch.arange(args.seq_len).expand(args.batch_size, -1))
    # Save the data
    save_dir = os.path.join("data", str(date.today()))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "seqs.pt"), "wb") as f:
        pickle.dump(seqs, f)
    with open(os.path.join(save_dir, "seqs_ans_pos_start.pt"), "wb") as f:
        pickle.dump(seqs_ans_pos_start, f)
    with open(os.path.join(save_dir, "seqs_ans_pos_end.pt"), "wb") as f:
        pickle.dump(seqs_ans_pos_end, f)
    args.special_tokens = dict(args.special_tokens)
    OmegaConf.save(args, os.path.join(save_dir, "args.yaml"))
    


