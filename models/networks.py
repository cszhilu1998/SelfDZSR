import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
from util.util import extract_image_patches
import torchvision.ops as ops


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer,
										step_size=opt.lr_decay_iters,
										gamma=0.5)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
												   mode='min',
												   factor=0.2,
												   threshold=0.01,
												   patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
												   T_max=opt.niter,
												   eta_min=0)
	else:
		return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	def init_func(m):  # define the initialization function
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
				or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			elif init_type == 'uniform':
				init.uniform_(m.weight.data, b=init_gain)
			else:
				raise NotImplementedError('[%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	if init_type != 'default' and init_type is not None:
		init_weights(net, init_type, init_gain=init_gain)
	return net

'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
	if len(args) == 1:
		args = args[0]
	if isinstance(args, nn.Module):
		return args
	modules = OrderedDict()
	if isinstance(args, OrderedDict):
		for k, v in args.items():
			modules[k] = seq(v)
		return nn.Sequential(modules)
	assert isinstance(args, (list, tuple))
	return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# ===================================
'''

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
		 output_padding=0, dilation=1, groups=1, bias=True,
		 padding_mode='zeros', mode='CBR'):
	L = []
	for t in mode:
		if t == 'C':
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=groups,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'X':
			assert in_channels == out_channels
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=in_channels,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels,
										out_channels=out_channels,
										kernel_size=kernel_size,
										stride=stride,
										padding=padding,
										output_padding=output_padding,
										groups=groups,
										bias=bias,
										dilation=dilation,
										padding_mode=padding_mode))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'i':
			L.append(nn.InstanceNorm2d(out_channels))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'S':
			L.append(nn.Sigmoid())
		elif t == 'P':
			L.append(nn.PReLU())
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return seq(*L)


class MeanShift(nn.Conv2d):
	""" is implemented via group conv """
	def __init__(self, rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040),
				 rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1, groups=3)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.ones(3).view(3, 1, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False


def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3,
						  stride=1, padding=1, bias=True, mode='2R'):
	# mode examples: 2, 2R, 2BR, 3, ..., 4BR.
	assert len(mode)<4 and mode[0] in ['2', '3', '4']
	up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size,
			   stride, padding, bias=bias, mode='C'+mode)
	return up1


class PatchSelect(nn.Module):
	def __init__(self,  stride=1):
		super(PatchSelect, self).__init__()
		self.stride = stride             

	def rgb2gray(self, rgb):
		r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
		gray = 0.2989*r + 0.5870*g + 0.1140*b
		return gray

	def forward(self, query, key):
		# query: lr, [N, C, H, W]
		# key:  ref, [N, C, 4*H, 4*W]
		# query = self.rgb2gray(query)
		# key = self.rgb2gray(key)

		shape_query = query.shape
		shape_key = key.shape
		
		P = shape_key[3] - shape_query[3] + 1  # patch number per row
		key = extract_image_patches(key, ksizes=[shape_query[2], shape_query[3]], 
				strides=[self.stride, self.stride], rates=[1, 1], padding='valid')
		# [N, C*H*W, P*P']

		key = key.view(shape_query[0], shape_query[1], shape_query[2] * shape_query[3], -1)
		sorted_key, _ = torch.sort(key, dim=2)

		query = query.view(shape_query[0], shape_query[1], shape_query[2] * shape_query[3], 1)
		sorted_query, _ = torch.sort(query, dim=2)
		y = torch.mean(torch.mean(torch.abs(sorted_key - sorted_query), 2), 1)  # [N, P*P']

		# query = query.view(shape_query[0], shape_query[1] * shape_query[2] * shape_query[3], 1)
		# y = torch.mean(torch.abs(key - query), 1)  # [N, P*P']
		relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True)  # [N, P*P']   
		
		return  hard_indices.view(-1), P # , relavance_maps


class AdaptBlock(nn.Module):  # Adaptive Spatial Transformer Networks (AdaSTN)
	def __init__(self, opt, inplanes=64, outplanes=64, stride=1, dilation=1, deformable_groups=64):
		super(AdaptBlock, self).__init__()
		self.opt = opt
		self.mask = True

		regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
									   [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
		self.register_buffer('regular_matrix', regular_matrix.float())

		self.concat = conv(inplanes*2, inplanes*2, groups=inplanes*2, mode='CL')
		self.concat2 = conv(inplanes*2, inplanes, groups=inplanes, mode='CL')

		self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 5, 1, 2, bias=True)
		self.translation_conv = nn.Conv2d(inplanes, 2, 5, 1, 2, bias=True)

		self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
			padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)
		self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True) 

	def forward(self, x, h_hr, rand=True):
		N, _, H, W = x.shape

		if rand and not self.opt.isTrain:
			offset_xx = - self.regular_matrix.transpose(1,0).reshape((18)).repeat(N, H, W, 1).permute(0,3,1,2)
			out = self.adapt_conv(x, offset_xx)        
			out = self.relu(out)
			return out

		x_h_hr = self.concat2(self.concat(torch.cat([x, h_hr], dim=1)))

		transform_matrix = self.transform_matrix_conv(x_h_hr)
		transform_matrix = transform_matrix.permute(0,2,3,1).reshape((N*H*W,2,2))
		offset = torch.matmul(transform_matrix, self.regular_matrix)
		offset = offset - self.regular_matrix
		offset = offset.transpose(1,2).reshape((N,H,W,18)).permute(0,3,1,2)

		translation = self.translation_conv(x_h_hr)
		offset[:,0::2,:,:] += translation[:,0:1,:,:]
		offset[:,1::2,:,:] += translation[:,1:2,:,:]

		if rand and self.opt.isTrain:
			offset_xx = - self.regular_matrix.transpose(1,0).reshape((18)).repeat(N, H, W, 1).permute(0,3,1,2)
			for i in range(N):
				rand_num = np.random.rand() 
				if rand_num < self.opt.dropout:
					offset[i] = offset_xx[i]
		
		out = self.adapt_conv(x, offset) 
		out = self.relu(out)
		return out


class ResBlock(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
				 padding=1, bias=True, mode='CRC', predict=False):
		super(ResBlock, self).__init__()
		assert in_channels == out_channels
		self.predict = predict
		if mode[0] in ['R','L']:
			mode = mode[0].lower() + mode[1:]

		self.res = conv(in_channels, out_channels, kernel_size,
						  stride, padding=padding, bias=bias, mode=mode)

		if self.predict:
			mlp = [conv(64, 4, 1, padding=0, mode='CR'),
				conv(4, 64, 1, padding=0, mode='C')]
			self.mlp = seq(mlp)
		
	def forward(self, x, p=None):
		x_in = x.clone()
		if self.predict and p != None:
			kernel = self.mlp(p)
			x_in = kernel * x_in + x_in
		res = self.res(x_in)
		return x + res


class Predictor(nn.Module):
	def __init__(self, opt):
		super(Predictor, self).__init__()
		self.scale = opt.scale
		self.mean = MeanShift()
		
		head = [conv(3+3, 64, 3, 1, mode='CR')]
		self.head = seq(head)

		predictor = [conv(128, 64, 3, 1, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 conv(64, 64, 3, 2, mode='CR'),
					 nn.AdaptiveAvgPool2d(1) ]
		self.predictor = seq(predictor)
	
	def forward(self, lr, hr, concat):
		up_lr = F.interpolate(lr, size=hr.shape[2:], mode='bilinear', align_corners=True)
		up_lr = self.mean(up_lr)
		hr = self.mean(hr)
		lr_hr_center = torch.cat([up_lr, hr], dim=1)
		
		h = self.head(lr_hr_center)
		concat_up = F.interpolate(concat, size=h.shape[2:], mode='bilinear', align_corners=True)
		input = torch.cat([h, concat_up], 1)

		out = self.predictor(input)
		return out


class CorrespondenceGeneration(nn.Module):
	def __init__(self,
				 patch_size=3,
				 stride=1):
		super(CorrespondenceGeneration, self).__init__()
		self.patch_size = patch_size
		self.stride = stride

	def index_to_flow(self, max_idx):
		device = max_idx.device
		# max_idx to flow
		h, w = max_idx.size()
		flow_w = max_idx % w
		flow_h = max_idx // w

		grid_y, grid_x = torch.meshgrid(
			torch.arange(0, h).to(device),
			torch.arange(0, w).to(device))
		grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
		grid.requires_grad = False
		flow = torch.stack((flow_w, flow_h),
						   dim=2).unsqueeze(0).float().to(device)
		flow = flow - grid  # shape:(1, w, h, 2)
		
		flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

		return flow

	def forward(self, dense_features):
		# print(dense_features['dense_features1'].shape)
		N, C, H, W = dense_features['dense_features1'].shape 
		offset_list=[]
		val_list = []
		for ind in range(N):
			feat_in = dense_features['dense_features1'][ind]
			feat_ref = dense_features['dense_features2'][ind]
			c, h, w = feat_in.size()
			# print(feat_in.size(), feat_ref.size())
			feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
			feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h, w)

			_max_idx, _max_val = feature_match_index(
				feat_in,
				feat_ref,
				patch_size=self.patch_size,
				input_stride=self.stride,
				ref_stride=self.stride,
				is_norm=True,
				norm_input=False)

			# offset map 
			offset = self.index_to_flow(_max_idx)
			# shift offset 
			shifted_offset = []
			for i in range(0, 3):
				for j in range(0, 3):
					flow_shift = tensor_shift(offset, (i, j))
					shifted_offset.append(flow_shift)
			shifted_offset = torch.cat(shifted_offset, dim=0)
			offset_list.append(shifted_offset)
			val_list.append(_max_val)

		# size: [b, 9, h, w, 2], the order of the last dim: [x, y]
		offset_list = torch.stack(offset_list, dim=0)
		pre_offset_reorder = torch.zeros([N, 18, H, W], device=offset_list.device)
		pre_offset_reorder[:, 0::2, :, :] = offset_list[:, :, :, :, 1]
		pre_offset_reorder[:, 1::2, :, :] = offset_list[:, :, :, :, 0]

		# val_list = torch.stack(val_list, dim=0).unsqueeze(1)
		# val_list = torch.nn.functional.pad(val_list, (1, 1, 1, 1))
		
		return pre_offset_reorder

def sample_patches(inputs, patch_size=3, stride=1):
	"""Extract sliding local patches from an input feature tensor.
	The sampled pathes are row-major.

	Args:
		inputs (Tensor): the input feature maps, shape: (c, h, w).
		patch_size (int): the spatial size of sampled patches. Default: 3.
		stride (int): the stride of sampling. Default: 1.

	Returns:
		patches (Tensor): extracted patches, shape: (c, patch_size,
			patch_size, n_patches).
	"""

	c, h, w = inputs.shape
	patches = inputs.unfold(1, patch_size, stride)\
					.unfold(2, patch_size, stride)\
					.reshape(c, -1, patch_size, patch_size)\
					.permute(0, 2, 3, 1)
	return patches


def feature_match_index(feat_input,
						feat_ref,
						patch_size=3,
						input_stride=1,
						ref_stride=1,
						is_norm=True,
						norm_input=False):
	"""Patch matching between input and reference features.

	Args:
		feat_input (Tensor): the feature of input, shape: (c, h, w).
		feat_ref (Tensor): the feature of reference, shape: (c, h, w).
		patch_size (int): the spatial size of sampled patches. Default: 3.
		stride (int): the stride of sampling. Default: 1.
		is_norm (bool): determine to normalize the ref feature or not.
			Default:True.

	Returns:
		max_idx (Tensor): The indices of the most similar patches.
		max_val (Tensor): The correlation values of the most similar patches.
	"""

	# patch decomposition, shape: (c, patch_size, patch_size, n_patches)
	patches_ref = sample_patches(feat_ref, patch_size, ref_stride)

	# normalize reference feature for each patch in both channel and
	# spatial dimensions.

	# batch-wise matching because of memory limitation
	_, h, w = feat_input.shape
	# batch_size = int(1024.**2 * 512 / (h * w))
	batch_size = int(1024.**2 * 1024 / (h * w))
	n_patches = patches_ref.shape[-1]

	# print(batch_size, n_patches)
	max_idx, max_val = None, None
	for idx in range(0, n_patches, batch_size):
		batch = patches_ref[..., idx:idx + batch_size]
		if is_norm:
			batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)
		corr = F.conv2d(
			feat_input.unsqueeze(0),
			batch.permute(3, 0, 1, 2),
			stride=input_stride)

		max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)

		if max_idx is None:
			max_idx, max_val = max_idx_tmp, max_val_tmp
		else:
			indices = max_val_tmp > max_val
			max_val[indices] = max_val_tmp[indices]
			max_idx[indices] = max_idx_tmp[indices] + idx

	# if norm_input:
	# 	patches_input = sample_patches(feat_input, patch_size, input_stride)
	# 	norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
	# 	norm = norm.view(
	# 		int((h - patch_size) / input_stride + 1),
	# 		int((w - patch_size) / input_stride + 1))
	# 	max_val = max_val / norm

	return max_idx, max_val


def tensor_shift(x, shift=(2, 2), fill_val=0):
	""" Tensor shift.

	Args:
		x (Tensor): the input tensor. The shape is [b, h, w, c].
		shift (tuple): shift pixel.
		fill_val (float): fill value

	Returns:
		Tensor: the shifted tensor.
	"""

	_, h, w, _ = x.size()
	shift_h, shift_w = shift
	new = torch.ones_like(x) * fill_val

	if shift_h >= 0 and shift_w >= 0:
		len_h = h - shift_h
		len_w = w - shift_w
		new[:, shift_h:shift_h + len_h,
			shift_w:shift_w + len_w, :] = x.narrow(1, 0,
												   len_h).narrow(2, 0, len_w)
	else:
		raise NotImplementedError
	return new

