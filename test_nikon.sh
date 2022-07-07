#!/bin/bash
echo "Start to test the model...."

dataroot="/Data/dataset/RealSR/RealSR_patches"
name="nikon_l1" # You can replace 'nikon_l1' with 'nikon_l1sw' when testing the model trained by SW loss.
scale='4'
data='dsr'
cam='False'
device="0"
iter="401"

python test.py \
    --model refsr  --name $name  --dataset_name $data  --chop True  --full_res True  --predict True \
    --load_iter $iter    --save_imgs True  --calc_psnr False  --gpu_ids $device  --scale $scale  --dataroot $dataroot

python metrics.py  --device $device --name $name --load_iter $iter  --cam $cam  --dataroot $dataroot
