#!/bin/bash


python -u train.py --save_path runs/megan_base_cloud \
	--clean_trainset data/clean_trainset \
	--noisy_trainset data/noisy_trainset \
    --clean_valset data/clean_valset \
    --noisy_valset data/noisy_valset \
    --epoch 50 \
    --batch_size 100 \
    --vanilla_gan \
    --genc_fmaps    64 128 256 512 1024 2048 \
    --genc_poolings 4  4   4   4   2    2    \
    --gdec_poolings 2  2   4   4   4    4    \
    --denc_fmaps    64 128 256 512 1024 2048 \
    --denc_poolings 4  4   4   4   2    2    \
	--no_train_gen --no_bias \
    --num_workers 4 --cache_dir data_tmp \

