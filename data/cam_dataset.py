import random
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from .imlib import imlib
from util.util import *
import torch
from .degrade.degrade_kernel import degrade_kernel


class CAMDataset(BaseDataset):  # CameraFusion dataset
	def __init__(self, opt, split='train', dataset_name='DSR'):
		super(CAMDataset, self).__init__(opt, split, dataset_name)
		if self.root == '':
			rootlist = ['']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break
	
		self.batch_size = opt.batch_size
		self.mode = opt.mode  # RGB, Y or L=
		self.finetune = opt.finetune
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.patch_size = opt.patch_size # 48

		self.scale = 2 # opt.scale
		self.camera = {'CameraFusion':['IMG']}

		self.x_scale = 'x' + str(opt.scale)
		if split == 'train':
			self.train_root = os.path.join(self.root, 'CameraFusion', 'train_')
			self.lr_dirs, self.hr_dirs, self.names = self._get_image_dir(self.train_root)
			self.len_data = 1000 * self.batch_size  # len(self.names)
			self._getitem = self._getitem_train

		elif split == 'val':
			self.val_root = os.path.join(self.root, 'CameraFusion', 'test_')
			self.lr_dirs, self.hr_dirs, self.names = self._get_image_dir(self.val_root) # self.camera[opt.camera]
			self._getitem = self._getitem_val
			self.len_data = len(self.names)

		elif split == 'test':
			self.test_root = os.path.join(self.root, 'CameraFusion', 'test_')
			self.lr_dirs, self.hr_dirs, self.names = self._get_image_dir(self.test_root) #, self.camera[opt.camera])
			self._getitem = self._getitem_test
			self.len_data = len(self.names)

		else:
			raise ValueError

		self.lr_images = [0] * len(self.names)
		self.hr_images = [0] * len(self.names)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		idx = idx % len(self.names)

		lr_img = self.lr_images[idx]
		hr_img = self.hr_images[idx]
		lr_img, hr_img, _ = self._crop_patch(lr_img, hr_img, p=self.patch_size)

		lr_img, hr_img = augment(lr_img, hr_img)

		lr_ref_img, hr_ref_img, crop_coord = self._crop_ref(lr_img, hr_img)

		hr_trans = np.transpose(hr_img, (1, 2, 0))

		hr_bili_noise, degradation_list = degrade_kernel(hr_trans, sf=2)
		noise = hr_bili_noise
		noise = np.transpose(noise, (2, 0, 1))
		
		noise = np.float32(noise) / 255
		lr_img = np.float32(lr_img) / 255 
		hr_img = np.float32(hr_img) / 255
		lr_ref_img = np.float32(lr_ref_img) / 255
		hr_ref_img = np.float32(hr_ref_img) / 255

		return {'lr': lr_img,
				'hr': hr_img,
				'noise': noise,
				'lr_ref': lr_ref_img,
				'hr_ref': hr_ref_img,
				'crop_coord': crop_coord,
				'fname': self.names[idx]}

	def _getitem_val(self, idx):
		lr_img = self.lr_images[idx]
		hr_img = self.hr_images[idx]
		lr_img, hr_img, _ = self._crop_center(lr_img, hr_img, p=256)
		lr_ref_img, hr_ref_img, crop_coord = self._crop_center(lr_img, hr_img)

		hr_trans = np.transpose(hr_img, (1, 2, 0))

		hr_bili_noise, degradation_list = degrade_kernel(hr_trans, sf=2)
		noise = hr_bili_noise
		noise = np.transpose(noise, (2, 0, 1))
		
		noise = np.float32(noise) / 255
		lr_img = np.float32(lr_img) / 255 
		hr_img = np.float32(hr_img) / 255
		lr_ref_img = np.float32(lr_ref_img) / 255
		hr_ref_img = np.float32(hr_ref_img) / 255

		return {'lr': lr_img,
				'hr': hr_img,
				'noise': noise,
				'lr_ref': lr_ref_img,
				'hr_ref': hr_ref_img,
				'crop_coord': crop_coord,
				'fname': self.names[idx]}

	def _getitem_test(self, idx):
		lr_img = self.lr_images[idx]
		hr_img = self.hr_images[idx]

		if not self.opt.full_res:
			lr_img, hr_img, _ = self._crop_center(lr_img, hr_img, p=400)
		lr_ref_img, hr_ref_img, crop_coord = self._crop_center(lr_img, hr_img)

		hr_trans = np.transpose(hr_img, (1, 2, 0))

		hr_bili_noise, degradation_list = degrade_kernel(hr_trans, sf=2)
		noise = hr_bili_noise
		noise = np.transpose(noise, (2, 0, 1))
		
		noise = np.float32(noise) / 255
		lr_img = np.float32(lr_img) / 255 
		hr_img = np.float32(hr_img) / 255
		lr_ref_img = np.float32(lr_ref_img) / 255
		hr_ref_img = np.float32(hr_ref_img) / 255

		return {'lr': lr_img,
				'hr': hr_img,
				'noise': noise,
				'lr_ref': lr_ref_img,
				'hr_ref': hr_ref_img,
				'crop_coord': crop_coord,
				'fname': self.names[idx]}
   
	def _get_image_dir(self, dataroot, cameras=['']):
		lr_dirs = []
		hr_dirs = [] # input_x4_raw target_x4_rgb
		image_names = []

		for file_name in os.listdir(dataroot + 'LR/'):  
			if cameras != ['']:
				for camera in cameras:
					if file_name.startswith(camera):
						lr_dirs.append(dataroot + 'LR/' + file_name)
						hr_file_name = file_name.replace('x1', self.x_scale)
						image_names.append(hr_file_name)
						hr_dirs.append(dataroot + 'HR/' + hr_file_name)
			else:
				lr_dirs.append(dataroot + 'LR/' + file_name)
				hr_file_name = file_name.replace('x1', self.x_scale)
				image_names.append(hr_file_name)
				hr_dirs.append(dataroot + 'HR/' + hr_file_name)
		image_names = sorted(image_names) 
		lr_dirs = sorted(lr_dirs) 
		hr_dirs = sorted(hr_dirs) 

		return lr_dirs, hr_dirs, image_names

	def _crop_patch(self, lr, hr, p):
		ih, iw = lr.shape[-2:]
		pw = random.randrange(0, iw - p + 1)
		ph = random.randrange(0, ih - p + 1)
		hpw, hph = self.scale * pw, self.scale * ph
		hr_patch_size = self.scale * p
		crop_coord = [ph, ph+p, pw, pw+p]
		crop_coord = np.array(crop_coord, dtype=np.int32)
		return lr[..., ph:ph+p, pw:pw+p], \
			   hr[..., hph:hph+hr_patch_size, hpw:hpw+hr_patch_size], \
			   crop_coord

	def _crop_ref(self, lr, hr, p=None):
		p = self.patch_size // self.scale # random.randrange(4, self.patch_size-16)
		ih, iw = lr.shape[-2:]
		pw = random.randrange(0, iw - p + 1)
		ph = random.randrange(0, ih - p + 1)
		hpw, hph = self.scale * pw, self.scale * ph
		hr_patch_size = self.scale * p
		crop_coord = [ph, ph+p, pw, pw+p]
		crop_coord = np.array(crop_coord, dtype=np.int32)
		return lr[..., ph:ph+p, pw:pw+p], \
			   hr[..., hph:hph+hr_patch_size, hpw:hpw+hr_patch_size], \
			   crop_coord

	def _crop_center(self, lr, hr, fw=0.5, fh=0.5, p=0):
		ih, iw = lr.shape[-2:]
		if p != 0:
			fw = p / iw
			fh = p / ih
		lr_patch_h, lr_patch_w = round(ih * fh), round(iw * fw)
		ph = ih // 2 - lr_patch_h // 2
		pw = iw // 2 - lr_patch_w // 2
		hph, hpw = self.scale * ph, self.scale * pw
		hr_patch_h, hr_patch_w = self.scale * lr_patch_h, self.scale * lr_patch_w
		crop_coord = [ph, ph+lr_patch_h, pw, pw+lr_patch_w]
		crop_coord = np.array(crop_coord, dtype=np.int32)
		return lr[..., ph:ph+lr_patch_h, pw:pw+lr_patch_w], \
			   hr[..., hph:hph+hr_patch_h, hpw:hpw+hr_patch_w],\
			   crop_coord

def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	i, obj = arg
	for _ in range(3):
		try:
			obj.lr_images[i] = obj.imio.read(obj.lr_dirs[i])
			obj.hr_images[i] = obj.imio.read(obj.hr_dirs[i])
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	from multiprocessing.dummy import Pool
	from tqdm import tqdm
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass