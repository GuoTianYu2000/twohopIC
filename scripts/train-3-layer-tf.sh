#! /bin/bash


iter_num=10000
hopk=2
layer=3
seq_len=24
lr=0.0003


python src/main_two_hop.py max_iters=${iter_num} task_name=multiHop fine_grid_log=${iter_num} seed=42 device_num=6 data_args.seq_len=${seq_len} num_data_workers=48 \
model_args.n_layers=${layer} model_args.dim=256 model_args.n_heads=1  data_args.hopk=${hopk} wandb_args.name=multi_hop_${hopk}_L${layer}\
optim_args.use_sgd=False optim_args.learning_rate=${lr} optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
save_dir=multi_hop_${hopk}_L${layer}_seq_len_${seq_len}_lr_${lr}
