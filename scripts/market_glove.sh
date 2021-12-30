#!/bin/bash
#--resume-i1  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I1100.pth.tar \
#--resume-i2  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I2100.pth.tar \
#--resume-x  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_x100.pth.tar \
#--resume-i2-sem  /media/data2/pengqy/TIP/SAL-master/log/market-results/checkpoint_ep_pt_I2sem100.pth.tar

#--resume-i1 /public/users/pengqy/log/market-results-finetune/checkpoint_ep_jt_I1100.pth.tar \
#--resume-i2 /public/users/pengqy/log/market-results-finetune/checkpoint_ep_jt_I2100.pth.tar \
#--resume-i2-sem /public/users/pengqy/log/market-results-finetune/checkpoint_ep_jt_I2sem100.pth.tar \
#--resume-g /public/users/pengqy/log/market-results-finetune/checkpoint_ep_jt_G100.pth.tar \
#--resume-x /public/users/pengqy/log/market-results-finetune/checkpoint_ep_jt_x100.pth.tar \
#--resume-a /public/users/pengqy/log/market-results-finetune/checkpoint_ep_jt_a100.pth.tar \
	   
# abort entire script on error
set -e

# train model
cd ..
python3 train_glove.py \
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
       --eval-freq 1 \
       --save-pt 100 \
       --adj-file  /media/data2/pengqy/TIP/dataset/adj_market.pkl \
       --save-dir /public/users/pengqy/log/temp \
	          --resume-i1  /public/users/pengqy/log/market/checkpoint_ep_pt_I1100.pth.tar \
       --resume-i2  /public/users/pengqy/log/market/checkpoint_ep_pt_I2100.pth.tar \
       --resume-x  /public/users/pengqy/log/market/checkpoint_ep_pt_x100.pth.tar \
       --resume-i2-sem  /public/users/pengqy/log/market/checkpoint_ep_pt_I2sem100.pth.tar \
       --gpu-devices 1 \
       --al-alone-lr 0.000001 \
       --sc-alone-lr 0.0001 \
       --al-alone-gamma 0.01 \
       --sc-alone-gamma 0.01 \
       --extra-num 0 \
       --t 0.9 \
       --extra-adj 0.0001 \
	   --evaluate
	  
