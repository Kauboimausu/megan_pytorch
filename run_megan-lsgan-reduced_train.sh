#!/bin/bash


python -u train.py --save_path runs/megan_lsgan_reduced_cloud \
	--clean_trainset data/clean_trainset \
	--noisy_trainset data/noisy_trainset \
    --clean_valset data/clean_valset \
    --noisy_valset data/noisy_valset \
    --epoch 50 \
    --max_samples 1500 \
    --batch_size 100 \
    --genc_fmaps    64 128 256 512 1024 \
    --genc_poolings 4  4   4   4   4    \
    --gdec_poolings 4  4   4   4   4    \
    --denc_fmaps    64 128 256 512 1024 \
    --denc_poolings 4  4   4   4   4    \
	--no_train_gen \
    --num_workers 8 --cache_dir data_tmp \

