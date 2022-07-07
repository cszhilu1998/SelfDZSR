import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L
from util.util import *
import sys
import torchvision.ops as ops
from .extractor_model import ContrasExtractorSep


class RefSRModel(BaseModel):
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		return parser

	def __init__(self, opt):
		super(RefSRModel, self).__init__(opt)

		self.opt = opt

		self.visual_names = ['data_lr', 'data_hr', 'data_sr'] 
		self.loss_names = ['RefSR_L1','RefSR_SWD','RefSR_Total','KernelGen_L1', 'KernelGen_Filter', 'KernelGen_Total']  #

		self.model_names = ['RefSR', 'KernelGen']  #
		self.optimizer_names = ['RefSR_optimizer_%s' % opt.optimizer, 'KernelGen_optimizer_%s' % opt.optimizer] #

		RefSR = SelfRefSR(opt)
		self.netRefSR = N.init_net(RefSR, opt.init_type, opt.init_gain, opt.gpu_ids)

		kernelgen = KernelGen(opt)
		self.netKernelGen= N.init_net(kernelgen, opt.init_type, opt.init_gain, opt.gpu_ids)

		student = ContrasExtractorSep()
		self.netStudent = N.init_net(student, opt.init_type, opt.init_gain, opt.gpu_ids)
		self.set_requires_grad(self.netStudent, requires_grad=False)

		if self.opt.scale == 4:
			self.load_network_path(self.netStudent, './ckpt/nikon_pretrain_models/Student_model_400.pth')
		if self.opt.scale == 2:
			self.load_network_path(self.netStudent, './ckpt/camerafusion_pretrain_models/Student_model_400.pth')

		if self.isTrain:
			if self.opt.scale == 4:
				self.load_network_path(self.netKernelGen, './ckpt/nikon_pretrain_models/KernelGen_model_400.pth')
			if self.opt.scale == 2:
				self.load_network_path(self.netKernelGen, './ckpt/camerafusion_pretrain_models/KernelGen_model_400.pth')

			self.optimizer_RefSR = optim.Adam(self.netRefSR.parameters(),
										  lr=opt.lr,
										  betas=(opt.beta1, opt.beta2),
										  weight_decay=opt.weight_decay)

			self.optimizer_KernelGen = optim.Adam(self.netKernelGen.parameters(),
								lr=opt.lr/2,
								betas=(opt.beta1, opt.beta2),
								weight_decay=opt.weight_decay)

			self.optimizers = [self.optimizer_RefSR, self.optimizer_KernelGen] 

			self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
			self.criterionSWD = N.init_net(L.SWDLoss(), gpu_ids=opt.gpu_ids)
			self.criterionFilter = N.init_net(L.FilterLoss(), gpu_ids=opt.gpu_ids)
		else:
			self.select = N.PatchSelect()

	def set_input(self, input):
		self.data_lr = input['lr'].to(self.device)
		self.data_hr = input['hr'].to(self.device)
		self.data_lr_ref = input['lr_ref'].to(self.device)
		self.data_hr_ref = input['hr_ref'].to(self.device)
		self.data_noise = input['noise'].to(self.device)
		self.crop_coord = input['crop_coord'].to(self.device)
		self.image_name = input['fname']

	def forward(self):		
		if self.opt.chop and not self.opt.isTrain:
			if self.opt.scale == 4:
				self.data_sr = self.forward_chop_x4(self.data_lr, self.data_hr_ref) 
			elif self.opt.scale == 2:
				self.data_sr = self.forward_chop_x2(self.data_lr, self.data_hr_ref)
		else:
			self.data_down_hr, self.weight = self.netKernelGen(self.data_hr, self.data_lr)
			self.data_down_hr_de = self.data_down_hr.detach()
	
			self.data_hr_bic = F.interpolate(self.data_hr, scale_factor=1/self.opt.scale, \
				mode='bicubic', align_corners=True)
			self.data_down_hr_de = self.data_down_hr_de + (self.data_noise - self.data_hr_bic) * 1.0
			self.data_down_hr_de = torch.clamp(self.data_down_hr_de, 0, 1)
			
			self.stu_out = self.netStudent(self.data_down_hr_de, self.data_hr_ref)

			self.data_sr = self.netRefSR(
				self.data_lr, self.data_down_hr_de, self.stu_out, self.data_hr_ref,
				self.data_lr_ref, self.data_hr_ref,  self.crop_coord, self.data_hr_ref)

	def backward(self):
		self.loss_KernelGen_L1 = self.criterionL1(self.data_lr, self.data_down_hr).mean()
		self.loss_KernelGen_Filter = self.criterionFilter(self.weight[0]).mean() * 100
		for conv_w in self.weight[1:]:
			self.loss_KernelGen_Filter = self.loss_KernelGen_Filter + self.criterionFilter(conv_w).mean() * 100
		self.loss_KernelGen_Total = self.loss_KernelGen_L1 + self.loss_KernelGen_Filter

		self.loss_RefSR_L1 = self.criterionL1(self.data_hr, self.data_sr).mean()
		self.loss_RefSR_SWD = self.criterionSWD(self.data_hr, self.data_sr).mean()
		self.loss_RefSR_Total = self.loss_RefSR_L1 + self.loss_RefSR_SWD

		self.loss_Total = self.loss_RefSR_Total + self.loss_KernelGen_Total
		self.loss_Total.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer_KernelGen.zero_grad()
		self.optimizer_RefSR.zero_grad()
		self.backward()
		self.optimizer_KernelGen.step()
		self.optimizer_RefSR.step()

	def forward_chop_x4(self, lr, ref):
		_, _, lr_h, lr_w = lr.shape
		new_lr_h = (lr_h // 24 + 1) * 24 + 16
		new_lr_w = (lr_w // 24 + 1) * 24 + 16

		pad_h = new_lr_h - lr_h
		pad_w = new_lr_w - lr_w

		pad_top = int(pad_h / 2.)
		pad_bottom = pad_h - pad_top
		pad_left = int(pad_w / 2.)
		pad_right = pad_w - pad_left

		paddings = (pad_left, pad_right, pad_top, pad_bottom)
		new_lr = torch.nn.ReflectionPad2d(paddings)(lr)
		# new_ref = torch.nn.ReflectionPad2d((pad_left-8, pad_right-8, pad_top-8, pad_bottom-8))(ref)
		# new_lr = torch.nn.ZeroPad2d(paddings)(lr)
		# lr = new_lr[:, :, pad_top:pad_top+lr_h, pad_left:pad_left+lr_w]
		num_h = num_w = 3
		patch_lr_h = (new_lr_h - 16) // num_h
		patch_lr_w = (new_lr_w - 16) // num_w

		sr_list = []
		ref_s = 8

		pre_lr = lr[:, :, 3*lr_h//8:5*lr_h//8, 3*lr_w//8:5*lr_w//8]
		pre_ref = ref
		# pad_left, pad_right, pad_top, pad_bottom = pad_left//4-2, pad_right//4-2, pad_top//4-2, pad_bottom//4-2
		# print(pad_left, pad_right, pad_top, pad_bottom)
		for j in range(num_h):
			for i in range(num_w):
				center_h = patch_lr_h * j + patch_lr_h // 2 + 8
				center_w = patch_lr_w * i + patch_lr_w // 2 + 8

				# print(center_h-patch_lr_h//2-8, center_h+patch_lr_h//2+8,
				#       center_w-patch_lr_w//2-8, center_w+patch_lr_w//2+8)
				patch_LR = new_lr[:, :, center_h-patch_lr_h//2-8:center_h+patch_lr_h//2+8, 
								  center_w-patch_lr_w//2-8:center_w+patch_lr_w//2+8]
				
				if i==1 and j==1:
					patch_ref = ref[:, :, center_h-patch_lr_h//2-8:center_h+patch_lr_h//2+8, 
								  center_w-patch_lr_w//2-8:center_w+patch_lr_w//2+8]
					# patch_ref_lr = lr[:, :, (center_h-patch_lr_h//2-8)//4:(center_h+patch_lr_h//2+8)//4, 
					# 			  (center_w-patch_lr_w//2-8)//4:(center_w+patch_lr_w//2+8)//4]
					
					round_h, round_w = round(patch_lr_h//8), round(patch_lr_w//8)
					crop_coord = [[round_h+6+pad_top//4, round_h+lr_h//4+6+pad_top//4, 
								   round_w+6+pad_left//4, round_w+lr_w//4+6+pad_left//4]]
					crop_coord = np.array(crop_coord, dtype=np.int32)
					# paste_ref = ref[:, :, lr_h%4//2:lr_h%4//2+lr_h//4*4, lr_w%4//2:lr_w%4//2+lr_w//4*4]
					paste_ref = ref[:, :, 0:lr_h//4*4, 0:lr_w//4*4]
					
					stu_out = self.netStudent(patch_LR, patch_ref)
					patch_sr = self.netRefSR(patch_LR, patch_LR, stu_out, patch_ref, pre_lr, pre_ref, crop_coord, paste_ref)
				else:
					lr_ = F.interpolate(patch_LR, scale_factor=1/ref_s, mode='bilinear', align_corners=True)
					ref_ = F.interpolate(ref, scale_factor=1/ref_s, mode='bilinear', align_corners=True)
			
					# i, P = self.select(self.netExtractor(lr_), self.netExtractor(ref_))
					idx, P = self.select(lr_, ref_)
					idx = idx.cpu()
					lr_ref_s = ref_s * 1

					ref_start_h = idx[0] // P * lr_ref_s
					ref_start_w = idx[0] % P * lr_ref_s
					if ref_start_h + patch_lr_h + 16 > ref.shape[2]:
						ref_start_h = ref.shape[2] - patch_lr_h - 16
					if ref_start_w + patch_lr_w + 16 > ref.shape[3]:
						ref_start_w = ref.shape[3] - patch_lr_w - 16	
					patch_ref = ref[:, :, ref_start_h:ref_start_h + patch_lr_h + 16, ref_start_w:ref_start_w + patch_lr_w + 16]
					# patch_ref_lr = lr[:, :, ref_start_h//4:(ref_start_h + patch_lr_h + 16)//4, 
					#                    ref_start_w//4:(ref_start_w + patch_lr_w + 16)//4]

					stu_out = self.netStudent(patch_LR, patch_ref)
					patch_sr = self.netRefSR(patch_LR, patch_LR, stu_out, patch_ref, pre_lr, pre_ref)

				sr_list.append(patch_sr[:, :, 8*4:8*4+patch_lr_h*4, 8*4:8*4+patch_lr_w*4])
		
		sr_list = torch.cat(sr_list, dim=0)
		sr_list = sr_list.view(sr_list.shape[0],-1)
		sr_list = sr_list.permute(1,0) 
		sr_list = torch.unsqueeze(sr_list, 0)
		output = F.fold(sr_list, output_size=(4*(new_lr_h-16), 4*(new_lr_w-16)), 
						kernel_size=(4*patch_lr_h, 4*patch_lr_w), padding=0, stride=(4*patch_lr_h, 4*patch_lr_w))
		sr_out = output[:, :, pad_top*4-32:pad_top*4+lr_h*4-32, pad_left*4-32:pad_left*4+lr_w*4-32]
	
		return sr_out

	def forward_chop_x2(self, lr, ref):
		_, _, lr_h, lr_w = lr.shape
		new_lr_h = (lr_h // 24 + 1) * 24 + 16
		new_lr_w = (lr_w // 24 + 1) * 24 + 16

		pad_h = new_lr_h - lr_h
		pad_w = new_lr_w - lr_w

		pad_top = int(pad_h / 2.)
		pad_bottom = pad_h - pad_top
		pad_left = int(pad_w / 2.)
		pad_right = pad_w - pad_left

		paddings = (pad_left, pad_right, pad_top, pad_bottom)
		new_lr = torch.nn.ReflectionPad2d(paddings)(lr)
		# print(2*(pad_left-8), 2*(pad_right-8), 2*(pad_top-8), 2*(pad_bottom-8))
		ref_pad = torch.nn.ReflectionPad2d(((pad_left-8), (pad_right-8), (pad_top-8), (pad_bottom-8)))(ref)
		# new_ref = torch.nn.ReflectionPad2d((pad_left-8, pad_right-8, pad_top-8, pad_bottom-8))(ref)
		# new_lr = torch.nn.ZeroPad2d(paddings)(lr)
		# lr = new_lr[:, :, pad_top:pad_top+lr_h, pad_left:pad_left+lr_w]
		num_h = num_w = 6
		patch_lr_h = (new_lr_h - 16) // num_h
		patch_lr_w = (new_lr_w - 16) // num_w

		sr_list = []
		ref_s = 8

		pre_lr = lr[:, :, lr_h//4:3*lr_h//4, lr_w//4:3*lr_w//4]
		pre_ref = ref
		# pad_left, pad_right, pad_top, pad_bottom = pad_left//4-2, pad_right//4-2, pad_top//4-2, pad_bottom//4-2
		# print(pad_left, pad_right, pad_top, pad_bottom)
		for j in range(num_h):
			for i in range(num_w):
				center_h = patch_lr_h * j + patch_lr_h // 2 + 8
				center_w = patch_lr_w * i + patch_lr_w // 2 + 8

				# print(center_h-patch_lr_h//2-8, center_h+patch_lr_h//2+8,
				#       center_w-patch_lr_w//2-8, center_w+patch_lr_w//2+8)
				patch_LR = new_lr[:, :, center_h-patch_lr_h//2-8:center_h+patch_lr_h//2+8, 
								  center_w-patch_lr_w//2-8:center_w+patch_lr_w//2+8]
				
				if (i==2 or i==3) and (j==2 or j==3):
					crop_coord = [[0, patch_lr_h+16, 0, patch_lr_w+16]]
					crop_coord = np.array(crop_coord, dtype=np.int32)
					ref_h, ref_w = ref_pad.shape[-2:]

					paste_ref = ref_pad[:, :, (j-1)*ref_h//3-patch_lr_h-16:(j-1)*ref_h//3+patch_lr_h+16, \
						(i-1)*ref_w//3-patch_lr_w-16:(i-1)*ref_w//3+patch_lr_w+16]
					
					patch_ref = paste_ref[..., 0:patch_lr_h+16, 0:patch_lr_w+16]  # any patch in paste_ref
					# print(patch_LR.shape, patch_ref.shape, paste_ref.shape)
					stu_out = self.netStudent(patch_LR, patch_ref)
					patch_sr = self.netRefSR(patch_LR, patch_LR, stu_out, patch_ref, pre_lr, pre_ref, crop_coord, paste_ref)

				# if i==0 or i==5 or i==1 or i==4 or j==0 or j==5 or j==1 or j==4:
				else:
					lr_ = F.interpolate(patch_LR, scale_factor=1/ref_s, mode='bilinear', align_corners=True)
					ref_ = F.interpolate(ref, scale_factor=1/ref_s, mode='bilinear', align_corners=True)
			
					idx, P = self.select(lr_, ref_)
					idx = idx.cpu()
					lr_ref_s = ref_s * 1

					ref_start_h = idx[0] // P * lr_ref_s
					ref_start_w = idx[0] % P * lr_ref_s
					if ref_start_h + patch_lr_h + 16 > ref.shape[2]:
						ref_start_h = ref.shape[2] - patch_lr_h - 16
					if ref_start_w + patch_lr_w + 16 > ref.shape[3]:
						ref_start_w = ref.shape[3] - patch_lr_w - 16	
					patch_ref = ref[:, :, ref_start_h:ref_start_h + patch_lr_h + 16, ref_start_w:ref_start_w + patch_lr_w + 16]

					stu_out = self.netStudent(patch_LR, patch_ref)
					patch_sr = self.netRefSR(patch_LR, patch_LR, stu_out, patch_ref, pre_lr, pre_ref)

				sr_list.append(patch_sr[:, :, 8*2:8*2+patch_lr_h*2, 8*2:8*2+patch_lr_w*2])
		
		sr_list = torch.cat(sr_list, dim=0)
		sr_list = sr_list.view(sr_list.shape[0],-1)
		sr_list = sr_list.permute(1,0) 
		sr_list = torch.unsqueeze(sr_list, 0)
		output = F.fold(sr_list, output_size=(2*(new_lr_h-16), 2*(new_lr_w-16)), 
						kernel_size=(2*patch_lr_h, 2*patch_lr_w), padding=0, stride=(2*patch_lr_h, 2*patch_lr_w))
		sr_out = output[:, :, pad_top*2-16:pad_top*2+lr_h*2-16, pad_left*2-16:pad_left*2+lr_w*2-16]
	
		return sr_out


class SelfRefSR(nn.Module): # DZSR Model
	def __init__(self, opt):
		super(SelfRefSR, self).__init__()
		self.opt = opt
		self.scale = opt.scale
		self.n_resblock = 16
		self.paste = opt.paste
		self.predict = opt.predict

		n_upscale = int(math.log(opt.scale, 2))
		n_feats = 64

		self.corr = N.CorrespondenceGeneration()

		ref_extractor = [N.MeanShift(),
						 N.conv(3, 64, mode='CR'),
						 N.conv(64, 64, mode='CRCRCRC')]
		self.ref_extractor = N.seq(ref_extractor)

		if self.paste:
			ref_head = [N.MeanShift(),
						DownBlock(self.scale),
						N.conv(3*self.scale**2, n_feats, mode='C')]
			self.ref_head = N.seq(ref_head)

		m_head = [N.MeanShift(),
				  N.conv(3, n_feats, mode='C')]
		self.head = N.seq(m_head)

		self.ada1 = N.AdaptBlock(opt, n_feats, n_feats)
		self.ada2 = N.AdaptBlock(opt, n_feats, n_feats)
		self.ada3 = N.AdaptBlock(opt, n_feats, n_feats)

		self.deform_conv = ops.DeformConv2d(64, 64, kernel_size=3, stride=1, \
			padding=1, dilation=1, bias=True, groups=64)
		self.ref_ada = N.AdaptBlock(opt, n_feats, n_feats)

		if self.predict:
			self.predictor = N.Predictor(opt)

		self.concat_fea = nn.Sequential(
			nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.1, True))
			
		for i in range(self.n_resblock):
			setattr(self, 'block%d'%i, N.ResBlock(n_feats, n_feats, mode='CRC', predict=self.predict))

		self.body_lastconv = N.conv(n_feats, n_feats, mode='C')

		if opt.scale == 3:
			m_up = N.upsample_pixelshuffle(n_feats, n_feats, mode='3')
		else:
			m_up = [N.upsample_pixelshuffle(n_feats, n_feats, mode='2') \
					  for _ in range(n_upscale)]
		self.up = N.seq(m_up)

		m_tail = [N.conv(n_feats, 3, mode='C'),
				  N.MeanShift(sign=1)]
		self.tail = N.seq(m_tail)

	def forward(self, lr, hr_bic, stu_out, data_hr_ref, pre_lr, pre_hr, crop_coord=None, paste_ref=None):
		N, C, H, W = lr.size()

		pre_offset = self.corr(stu_out)

		# pre_offset = F.interpolate(pre_offset, data_hr_ref.shape[-2:], mode='bilinear', align_corners=True) * 4
		pre_offset = F.interpolate(pre_offset, scale_factor=4, mode='bilinear', align_corners=True) * 4
		img_ref_feat = self.ref_extractor(data_hr_ref)
		ref_deform = self.deform_conv(img_ref_feat, pre_offset)    

		h = self.head(lr) 
		if hr_bic is None:
			h_hr = h.clone()
		else:
			h_hr = self.head(hr_bic) 

		h = self.ada1(h, h_hr)
		h = self.ada2(h, h_hr)
		h = self.ada3(h, h_hr)

		if self.paste:
			if paste_ref is not None and self.opt.isTrain:
				head_ref = self.ref_head(paste_ref)
				for i in range(N):
					rand_num = np.random.rand() # .to(offset_xx.device)[0]
					if rand_num < 0.3:
						ref_deform[i, :, crop_coord[i,0]:crop_coord[i,1], 
								crop_coord[i,2]:crop_coord[i,3]] = head_ref[i] # x_mid[i] head_ref[i]
			elif paste_ref is not None and not self.opt.isTrain:
				head_ref = self.ref_head(paste_ref)
				for i in range(N):
					ref_deform[i, :, crop_coord[i,0]:crop_coord[i,1], 
								crop_coord[i,2]:crop_coord[i,3]] = head_ref[i] # x_mid[i] head_ref[i]

		ref_deform = self.ref_ada(ref_deform, h_hr, rand=False) 
		cat_fea = self.concat_fea(torch.cat([h, ref_deform], 1))

		res = cat_fea.clone()
		
		if self.predict:
			pre = self.predictor(pre_lr, pre_hr, cat_fea)
		else:
			pre = None

		for i in range(self.n_resblock):
			res = getattr(self, 'block%d'%i)(res, pre)
		
		res = self.body_lastconv(res)
		res += h
		res = self.up(res)
		out = self.tail(res)	
		return out

class KernelGen(nn.Module):  # Auxiliary-LR Generator
	def __init__(self, opt):
		super(KernelGen, self).__init__()
		self.opt = opt
		n_feats = 64

		self.head_mean = N.MeanShift()

		self.down = DownBlock(scale=self.opt.scale)
		self.head = N.conv(3*self.opt.scale**2, n_feats, kernel_size=1, stride=1, padding=0, mode='CR')

		self.conv_7x7 = N.conv(n_feats, n_feats, kernel_size=7, stride=1, padding=3, mode='CR')
		self.conv_5x5 = N.conv(n_feats, n_feats, kernel_size=5, stride=1, padding=2, mode='CR')
		self.conv_3x3 = N.conv(n_feats, n_feats, kernel_size=3, stride=1, padding=1, mode='CR')

		self.conv_1x1 = N.conv(n_feats, n_feats, kernel_size=1, stride=1, padding=0, mode='CRCRCR')

		self.tail = N.conv(n_feats, 3, kernel_size=1, stride=1, padding=0, mode='C')

		self.guide_net = N.seq(
			N.conv(3*self.opt.scale**2+3, n_feats, 7, stride=2, padding=0, mode='CR'),
			N.conv(n_feats, n_feats, kernel_size=3, stride=1, padding=1, mode='CRCRC'),
			nn.AdaptiveAvgPool2d(1),
			N.conv(n_feats, n_feats, 1, stride=1, padding=0, mode='C')
		)

		self.tail_mean = N.MeanShift(sign=1)

	def forward(self, hr, lr):
		hr = self.head_mean(hr)
		lr = self.head_mean(lr)

		hr_down = self.down(hr)
		guide = self.guide_net(torch.cat([hr_down, lr], dim=1))

		head = self.head(hr_down)
		out = head * guide + head

		out = self.conv_3x3(self.conv_5x5(self.conv_7x7(out)))
		out = self.conv_1x1(out) + head
		out = self.tail(out)

		out = self.tail_mean(out)
		return out, [self.conv_7x7[0].weight, self.conv_5x5[0].weight, self.conv_3x3[0].weight]

class DownBlock(nn.Module):
	def __init__(self, scale):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		n, c, h, w = x.size()
		x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
		x = x.view(n, c*(self.scale**2), h//self.scale, w//self.scale)
		return x
