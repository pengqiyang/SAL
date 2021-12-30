#!/bin/bash
#--resume-i1  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I1100.pth.tar \
#--resume-i2  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I2100.pth.tar \
#--resume-x  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_x100.pth.tar \
#--resume-i2-sem  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I2sem100.pth.tar \
# abort entire script on error
set -e

# train model
cd ..
python3 train_finetune.py \
       -s Market-1501 \
       --root ../dataset/ \
       --optim adam \
       --label-smooth \
       --max-epoch-pt 0 \
       --max-epoch-jt 80 \
       --max-epoch-al 60 \
       --stepsize 20 40 60 80   \
       --stepsize-sal 20 40 \
       --train-batch-size 128 \
       --test-batch-size 100 \
       -a resnet50 \
       --save-dir /public/users/pengqy/log/market-finetune-result \
       --eval-freq 1 \
       --save-pt 100 \
       --adj-file  /media/data2/pengqy/TIP/dataset/adj_market.pkl \
       --resume-i1  /public/users/pengqy/log/market/checkpoint_ep_pt_I1100.pth.tar \
       --resume-i2  /public/users/pengqy/log/market/checkpoint_ep_pt_I2100.pth.tar \
       --resume-x  /public/users/pengqy/log/market/checkpoint_ep_pt_x100.pth.tar \
       --resume-i2-sem  /public/users/pengqy/log/market/checkpoint_ep_pt_I2sem100.pth.tar \
       --gpu-devices 1,2 \
       --t 0.9 \
       --adj-init 0.001 


