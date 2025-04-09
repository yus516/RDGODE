#!/bin/bash

# python train_with_weekday_seq.py --data=seattleloopweekdayweekend --device=cuda:0 --predicting_point=-1 --filter=0 --learning_rate=0.005 --weight_decay=0.001 --epochs=2000 --enable_bias=True --num_sequence=13 --num_weekday=60 --start_runs=0 --runs=6\
#     > logs/seattle-loop-semi-continuous-seq-pred-02102023-60-weekday-with-bias-0.005-0.001-adjoint-13-good.log 2>&1 &

# python train_with_weekday_seq.py --data=pemsbayweekdayweekend --device=cuda:1 --predicting_point=-1 --filter=0 --learning_rate=0.005 --weight_decay=0.001 --epochs=2000 --enable_bias=True --num_sequence=13 --num_weekday=12 --start_runs=0 --runs=6\
#     > logs/pems-bay-last-semi-continuous-pure-sigmoid-matmul-good-gate-seq-pred-03112023-12-weekday-with-bias-0.005-0.001-adjoint-13.log 2>&1 &

# python train_with_weekday_seq.py --data=pemsbayweekdayweekend --device=cuda:1 --predicting_point=-1 --filter=0 --learning_rate=0.005 --weight_decay=0.001 --epochs=2000 --enable_bias=True --num_sequence=13 --num_weekday=12 --start_runs=1 --runs=1 --num_rd_kernels=1\
#     > logs/pems-bay-start-from-last-semi-continuous-3-mod-matmul-softmax-all_difference-good-gate-seq-pred-03272023-12-weekday-with-bias-0.005-0.001-adjoint-13-[4-8].log 2>&1 &

# python train_with_weekday_seq.py --data=pemsbayweekdayweekend --device=cuda:1 --predicting_point=-1 --filter=0 --learning_rate=0.005 --weight_decay=0.001 --epochs=2000 --enable_bias=True --num_sequence=13 --num_weekday=12 --start_runs=1 --runs=1 --num_rd_kernels=8\
#     > logs/pems-bay-semi-continuous-binary-8mod-scale-matmul-good-gate-zero-high-low-seq-pred-03192023-first-12-weekday-with-bias-0.005-0.001-adjoint-13-[4-8].log 2>&1 &

# python train_with_weekday_seq.py --data=metrlaweekdayweekend --device=cuda:0 --predicting_point=-1 --filter=0 --learning_rate=0.005 --weight_decay=0.001 --epochs=2000 --enable_bias=True --num_sequence=13 --num_weekday=12 --start_runs=0 --runs=6 --num_rd_kernels=5\
#     > logs/metr-la-semi-continuous-softmax-5-kernel-seq-mse-pred-03272023-last-12-weekday-with-bias-0.005-0.001-adjoint-13.log 2>&1 &

python train_with_weekday_seq.py --data=metrlaweekdayweekend --device=cuda:0 --predicting_point=-1 --filter=0 --learning_rate=0.005 --weight_decay=0.001 --epochs=2000 --enable_bias=True --num_sequence=13 --num_weekday=36 --start_runs=0 --runs=6 --num_rd_kernels=2\
    > 04192024/metr-la-04192024-multi-step-only-jump-semi-continuous-softmax-2-kernel-seq-mae-pred-03272023-first-36-weekday-with-bias-0.005-0.001-adjoint-13.log 2>&1 &