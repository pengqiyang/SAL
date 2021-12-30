#!/bin/bash
#--resume-i1  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I1100.pth.tar \
#--resume-i2  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I2100.pth.tar \
#--resume-x  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_x100.pth.tar \
#--resume-i2-sem  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I2sem100.pth.tar \
# abort entire script on error
set -e

# train model

python3 test.py \
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
       --test-batch-size 1 \
       -a resnet50 \
       --eval-freq 1 \
       --save-pt 100 \
       --adj-file  /media/data2/pengqy/TIP/dataset/adj_market.pkl \
       --resume-i1  /public/users/pengqy/log/market/checkpoint_ep_pt_I1100.pth.tar \
       --resume-i2  /public/users/pengqy/log/market/checkpoint_ep_pt_I2100.pth.tar \
       --resume-x  /public/users/pengqy/log/market/checkpoint_ep_pt_x100.pth.tar \
       --resume-i2-sem  /public/users/pengqy/log/market/checkpoint_ep_pt_I2sem100.pth.tar \
       --save-dir log/temp \
       --gpu-devices 1 \
       --al-alone-lr 0.000001 \
       --sc-alone-lr 0.0001 \
       --al-alone-gamma 0.01 \
       --sc-alone-gamma 0.01 \
       --extra-num 0 \
       --t 0.9 \
       --extra-adj 0.0001 \
	   --evaluate


