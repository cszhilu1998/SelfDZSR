
echo "Start to train the model...."

dataroot="/Data/dataset/RealSR/RealSR_patches"
name="nikon_l1sw_try" 
scale='4'
data='dsr'

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
		mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --model refsr   --niter 401  --lr_decay_iters 200   --name $name   --dataroot $dataroot  --dataset_name $data  --print_freq 100 \
    --predict True  --save_imgs False   --scale $scale  --dropout 0.3  --calc_psnr True   --gpu_ids 0           -j 4  | tee $LOG   

