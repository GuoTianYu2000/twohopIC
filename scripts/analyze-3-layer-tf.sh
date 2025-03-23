#!/bin/bash

python src/dissecting.py date=2025-01-13 layer=3 head=1 hopk=2 steps=10000 run_path=runs/multi_hop_2_L3_seq_len_24_lr_0.0003
python src/dynamics.py date=2025-01-13 layer=3 head=1 hopk=2 steps=10000 run_path=runs/multi_hop_2_L3_seq_len_24_lr_0.0003
