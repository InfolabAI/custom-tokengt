#!/usr/bin/env bash

ulimit -c unlimited

python /compuworks/anaconda3/envs/py39/lib/python3.9/site-packages/fairseq_cli/train.py \
--user-dir ../tokengt \
--distributed-world-size 1 \
--num-workers 1 \
--dataset-name pcqm4mv2 \
--dataset-source ogb \
--task graph_prediction \
--criterion l1_loss \
--arch tokengt_base \
--orf-node-id \
--orf-node-id-dim 64 \
--stochastic-depth \
--prenorm \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 128 \
--fp16 \
--data-buffer-size 20 \
--save-dir ./ckpts/pcqv2-tokengt-orf64 \
--tensorboard-logdir ./tb/pcqv2-tokengt-orf64 \
--no-epoch-checkpoints
