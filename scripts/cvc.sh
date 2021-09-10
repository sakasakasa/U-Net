#!/bin/sh

#PBS -q h-regular
#PBS -l select=1
#PBS -W group_list=gk36
#PBS -l walltime=48:00:00

if [ -f ~/.bashrc ]; then
        . ~/.bashrc
fi

conda init
conda activate /lustre/gk36/k36062/sandbox/envs
module load pytorch/1.8.1
export PYTHONPATH="${HOME}/sandbox/pytorch181/lib/python3.9/site-packages"

cd /lustre/gk36/k36062/experiment/Pytorch-UNet-master-CVC
#python train.py --e 300 --s 0.5 --b 8
python train.py
