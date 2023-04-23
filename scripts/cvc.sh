#!/bin/bash
#PJM -g gk36
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=6:00:00
#PJM --fs /work,/data
#PJM -j 
#PJM -N LN-slimmable

module load cuda/11.1
module load pytorch/1.8.1
module load cudnn/8.1.0
module load nccl/2.8.4
source $PYTORCH_DIR/bin/activate

cd /work/gk36/k36062/k36062/experiment/Pytorch-UNet-master-Slim
python train.py --e 100 --s 0.5 --b 8 --g 1.0
#python train.py
