#!/bin/bash
echo "Start to test the model...."

dataroot="/Data/dataset/RealSR/RealSR_patches"
name="camerafusion_l1" # You can replace 'camerafusion_l1' with 'camerafusion_l1sw' when testing the model trained by SW loss.
scale='2'
data='cam'
cam='True'
device="0"
iter="401"

python test.py \
    --model refsr  --name $name  --dataset_name $data  --chop True  --full_res True  --predict True \
    --load_iter $iter    --save_imgs True  --calc_psnr False  --gpu_ids $device  --scale $scale  --dataroot $dataroot

python metrics.py  --device $device --name $name --load_iter $iter  --cam $cam  --dataroot $dataroot
