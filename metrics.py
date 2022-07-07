import random
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset
from scipy import misc
import imageio
import cv2
import torch
import math
from tqdm import tqdm
import lpips
import glob
from skimage.metrics import structural_similarity as ssim
from skimage import exposure
import argparse


def calc_psnr_np(sr, hr, range=255.):
	# shave = 2
	diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
	# diff = diff[shave:-shave, shave:-shave, :]
	total_mse = np.power(diff, 2)
	total_psnr = -10 * math.log10(total_mse.mean())

	return total_psnr

def calc_psnr_corner(sr, hr, range=255.):
	# shave = 2
	diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
	total_mse = np.power(diff, 2).mean()
	return total_mse

def lpips_norm(img):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img).to(device)

def calc_lpips(x_mask_out, x_canon, loss_fn_alex_1, loss_fn_alex_0=None):
	lpips_mask_out = lpips_norm(x_mask_out)
	lpips_canon = lpips_norm(x_canon)
	# LPIPS_0 = loss_fn_alex_0(lpips_mask_out, lpips_canon)
	LPIPS_1 = loss_fn_alex_1(lpips_mask_out, lpips_canon)
	return LPIPS_1.detach().cpu() #, LPIPS_1.detach().cpu()

def crop_part(out, ref, s):
	H, W, _ = out.shape
	c_H, c_W = (H - H//s)//2, (W - W//s)//2
	top = [out[:, 0:c_W,...], ref[:, 0:c_W,...]]
	left = [out[:c_H, c_W:c_W+W//s,...], ref[:c_H, c_W:c_W+W//s,...]]
	right = [out[c_H+H//s:, c_W:c_W+W//s, ...], ref[c_H+H//s:, c_W:c_W+W//s, ...]]
	bottom = [out[:, c_W+W//s:,...], ref[:, c_W+W//s:,...]]

	center = [out[c_H:c_H+H//s, c_W:c_W+W//s,...], ref[c_H:c_H+H//s, c_W:c_W+W//s,...]]
	return top, left, right, bottom, center

def calc_metrics(out, ref, s):
	parts = crop_part(out, ref, s)
	psnr_corner = 0.0
	ssim_corner = 0.0
	lpips_corner = 0.0
	areas = 0.0
	for part in parts[:-1]:
		total_psnr = calc_psnr_corner(part[0], part[1])
		SSIM = ssim(part[0], part[1], win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
		LPIPS_1 = calc_lpips(part[0], part[1], loss_fn_alex_1)
		area = part[0].shape[0] * part[0].shape[1]
		psnr_corner += total_psnr * area
		ssim_corner += SSIM 
		lpips_corner += LPIPS_1
		areas += area
	psnr_corner, ssim_corner, lpips_corner = - 10 * math.log10(psnr_corner / areas), ssim_corner / 4, lpips_corner / 4
	center_psnr = calc_psnr_np(parts[-1][0], parts[-1][1])
	center_SSIM = ssim(parts[-1][0], parts[-1][1], win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
	center_lpips = calc_lpips(parts[-1][0], parts[-1][1], loss_fn_alex_1)

	total_psnr = calc_psnr_np(out, ref)
	total_ssim = ssim(out, ref, win_size=11, data_range=255, multichannel=True, gaussian_weights=True)
	total_lpips = calc_lpips(out, ref, loss_fn_alex_1)

	return [psnr_corner, center_psnr, total_psnr, ssim_corner, center_SSIM, total_ssim, lpips_corner,  center_lpips, total_lpips]

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Test for argparse')
	parser.add_argument('--name', '-n', help='name')
	parser.add_argument('--device', default="0")
	parser.add_argument('--load_iter', default="401")
	parser.add_argument('--full_res', type=str2bool, default=True)
	parser.add_argument('--cam', type=str2bool, default=False)
	parser.add_argument('--dataroot', type=str, default='')
	args = parser.parse_args()

	print(args)

	args.device = "cuda:" + args.device
	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	loss_fn_alex_1 = lpips.LPIPS(net='alex', version='0.1').to(device)

	files = [
		'./ckpt/' + args.name + '/'
	]

	if args.full_res:
		if args.cam:
			s = 2
			ori_target = args.dataroot + '/CameraFusion/test_HR/'
			camera_names = {'iphone': ['IMG']}
		else:
			s = 4
			ori_target = args.dataroot + '/Nikon/test_HR/'
			# camera_names = {'canon': ['IMG', 'Canon', '0510'], 
			# 	'sony': ['sony'], 
			# 	'nikon': ['DSC'], 
			# 	'oly': ['P11'], 
			# 	'pan': ['pan'] }
			camera_names = {'nikon': ['DSC']}
	else:
		print('Error, please set --full_res \'True\'.')

	for file in files:
		if args.full_res:
			log_dir = '%s/log_full_%s.txt' % (file, args.load_iter)
		else:
			log_dir = '%s/log_patch_%s.txt' % (file, args.load_iter)
		f = open(log_dir, 'a')

		for camera in sorted(camera_names.keys()):
			names = []
			for file_name in os.listdir(ori_target):  
				for i in camera_names[camera]:
					if file_name.startswith(i):
							names.append(file_name)
			if names == []:
				continue
			names = sorted(names)
			f.write('\n=============%s=============\n' % (camera))
			print('\n=============%s=============\n' % (camera))
		
			ori_metrics = np.zeros([len(names), 9])
			i = 0
			for name in tqdm(names): 
				if args.full_res:
					pre_out = cv2.imread(file + 'sr_full_' + args.load_iter + '/' + name)[..., ::-1]
				else:
					pre_out = cv2.imread(file + 'sr_patch_' + args.load_iter + '/' + name)[..., ::-1]
					# pre_out = cv2.imread(file + 'sr_full_400/' + name)[..., ::-1]
				out = pre_out

				pre_ref = cv2.imread(ori_target + name)[..., ::-1]
				ref = pre_ref

				ori_metrics[i] = calc_metrics(out, ref, s)
				f.write('name: %s, \n corner_psnr: %.2f, \t center_psnr: %.2f, \t total_psnr: %.2f, \
									\n corner_SSIM: %.4f, \t center_SSIM: %.4f, \t total_SSIM: %.4f, \
									\n corner_LPIPS: %.3f, \t center_LPIPS: %.3f, \t total_LPIPS: %.3f \t \n' \
							% (name, ori_metrics[i][0], ori_metrics[i][1], ori_metrics[i][2], ori_metrics[i][3], ori_metrics[i][4], 
							ori_metrics[i][5], ori_metrics[i][6], ori_metrics[i][7], ori_metrics[i][8]))
				print('name: %s, \n corner_psnr: %.2f, \t center_psnr: %.2f, \t total_psnr: %.2f, \
									\n corner_SSIM: %.4f, \t center_SSIM: %.4f, \t total_SSIM: %.4f, \
									\n corner_LPIPS: %.3f, \t center_LPIPS: %.3f, \t total_LPIPS: %.3f \t \n' \
							% (name, ori_metrics[i][0], ori_metrics[i][1], ori_metrics[i][2], ori_metrics[i][3], ori_metrics[i][4], 
							ori_metrics[i][5], ori_metrics[i][6], ori_metrics[i][7], ori_metrics[i][8]))
				i = i + 1
			metrics_mean = np.mean(ori_metrics, axis=0)
			f.write('\n camera: %s ======  \
			         \n corner_psnr: %.2f, \t center_psnr: %.2f, \t total_psnr: %.2f, \
					 \n corner_SSIM: %.4f, \t center_SSIM: %.4f, \t total_SSIM: %.4f, \
					 \n corner_LPIPS: %.3f, \t center_LPIPS: %.3f, \t total_LPIPS: %.3f \t \n' \
					 % (camera, metrics_mean[0], metrics_mean[1], metrics_mean[2], 
					 metrics_mean[3], metrics_mean[4], metrics_mean[5], 
					 metrics_mean[6], metrics_mean[7], metrics_mean[8]))
			print('\n camera: %s ======  \
			         \n corner_psnr: %.2f, \t center_psnr: %.2f, \t total_psnr: %.2f, \
					 \n corner_SSIM: %.4f, \t center_SSIM: %.4f, \t total_SSIM: %.4f, \
					 \n corner_LPIPS: %.3f, \t center_LPIPS: %.3f, \t total_LPIPS: %.3f \t \n' \
					 % (camera, metrics_mean[0], metrics_mean[1], metrics_mean[2], 
					 metrics_mean[3], metrics_mean[4], metrics_mean[5], 
					 metrics_mean[6], metrics_mean[7], metrics_mean[8]))

		f.flush()
		f.close()