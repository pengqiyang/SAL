#!/bin/bash

# abort entire script on error
set -e

# train model
cd ..
python3 train_gcn.py \
-s PA100K  \
--root ../dataset/PA100K  \
--optim adam \
--label-smooth \
--max-epoch-jt 80   \
--max-epoch-pt 0   \
--max-epoch-al 60 \
--stepsize 20 40 60 80 \
--stepsize-sal 20 30 40 50  \
--train-batch-size 128  \
--test-batch-size 100  \
-a resnet50  \
--eval-freq 1 \
--save-pt 100 \
--adj-file  /media/data2/pengqy/TIP/dataset/PA100K/pa100k_adj.pkl \
--resume-i1  /public/users/pengqy/log/pa100k/checkpoint_ep_pt_I1100.pth.tar \
--resume-i2  /public/users/pengqy/log/pa100k/checkpoint_ep_pt_I2100.pth.tar \
--resume-x /public/users/pengqy/log/pa100k/checkpoint_ep_pt_x100.pth.tar \
--resume-i2-sem  /public/users/pengqy/log/pa100k/checkpoint_ep_pt_I2sem100.pth.tar \
--save-dir /public/users/pengqy/log/temp  \
--gpu-devices 2 \
--t 0.95 \
--adj-init 0.01 \
--al-alone-lr 0.000001 \
--sc-alone-lr 0.0001 \
--al-alone-gamma 0.01 \
--sc-alone-gamma 0.01 \
--extra-num 0 \
--evaluate

 





