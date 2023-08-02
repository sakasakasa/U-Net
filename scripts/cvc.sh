#!/bin/bash
#PJM -g gu14
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=48:00:00
#PJM --fs /work,/data
#PJM -j 
#PJM -N BN-switch-heapmap-cvc-dr

module load cuda/11.1
module load pytorch/1.8.1
module load cudnn/8.1.0
module load nccl/2.8.4
source $PYTORCH_DIR/bin/activate

cd /work/gk36/k36062/k36062/experiment/Pytorch-UNet-master-Slim
python train_cvc.py --e 100 --s 0.5 --b 8  --d "all" --BN "BN"

