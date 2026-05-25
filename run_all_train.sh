#!/bin/bash 
# Correr todos los scripts

set -e
mkdir -p runs

for script in run_megan-lsgan-reduced2_train.sh \
              run_megan-lsgan-reduced_train.sh \
              run_megan-lsgan_train.sh \
              run_megan-vanilla_train.sh; do
    bash $script
done