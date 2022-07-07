# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp, sqrt
from torch.nn import L1Loss, MSELoss
from torchvision import models
from util.util import grid_positions, warp


def normalize_batch(batch):
	mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
	std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
	return (batch - mean) / std
	
class VGG19(torch.nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		features = models.vgg19(pretrained=True).features
		self.relu1_1 = torch.nn.Sequential()
		self.relu1_2 = torch.nn.Sequential()

		self.relu2_1 = torch.nn.Sequential()
		self.relu2_2 = torch.nn.Sequential()

		self.relu3_1 = torch.nn.Sequential()
		self.relu3_2 = torch.nn.Sequential()
		self.relu3_3 = torch.nn.Sequential()
		self.relu3_4 = torch.nn.Sequential()

		self.relu4_1 = torch.nn.Sequential()
		self.relu4_2 = torch.nn.Sequential()
		self.relu4_3 = torch.nn.Sequential()
		self.relu4_4 = torch.nn.Sequential()

		self.relu5_1 = torch.nn.Sequential()
		self.relu5_2 = torch.nn.Sequential()
		self.relu5_3 = torch.nn.Sequential()
		self.relu5_4 = torch.nn.Sequential()

		for x in range(2):
			self.relu1_1.add_module(str(x), features[x])

		for x in range(2, 4):
			self.relu1_2.add_module(str(x), features[x])

		for x in range(4, 7):
			self.relu2_1.add_module(str(x), features[x])

		for x in range(7, 9):
			self.relu2_2.add_module(str(x), features[x])

		for x in range(9, 12):
			self.relu3_1.add_module(str(x), features[x])

		for x in range(12, 14):
			self.relu3_2.add_module(str(x), features[x])

		for x in range(14, 16):
			self.relu3_3.add_module(str(x), features[x])

		for x in range(16, 18):
			self.relu3_4.add_module(str(x), features[x])

		for x in range(18, 21):
			self.relu4_1.add_module(str(x), features[x])

		for x in range(21, 23):
			self.relu4_2.add_module(str(x), features[x])

		for x in range(23, 25):
			self.relu4_3.add_module(str(x), features[x])

		for x in range(25, 27):
			self.relu4_4.add_module(str(x), features[x])

		for x in range(27, 30):
			self.relu5_1.add_module(str(x), features[x])

		for x in range(30, 32):
			self.relu5_2.add_module(str(x), features[x])

		for x in range(32, 34):
			self.relu5_3.add_module(str(x), features[x])

		for x in range(34, 36):
			self.relu5_4.add_module(str(x), features[x])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		relu1_1 = self.relu1_1(x)
		relu1_2 = self.relu1_2(relu1_1)

		relu2_1 = self.relu2_1(relu1_2)
		relu2_2 = self.relu2_2(relu2_1)

		relu3_1 = self.relu3_1(relu2_2)
		relu3_2 = self.relu3_2(relu3_1)
		relu3_3 = self.relu3_3(relu3_2)
		relu3_4 = self.relu3_4(relu3_3)

		relu4_1 = self.relu4_1(relu3_4)
		relu4_2 = self.relu4_2(relu4_1)
		relu4_3 = self.relu4_3(relu4_2)
		relu4_4 = self.relu4_4(relu4_3)

		relu5_1 = self.relu5_1(relu4_4)
		relu5_2 = self.relu5_2(relu5_1)
		relu5_3 = self.relu5_3(relu5_2)
		relu5_4 = self.relu5_4(relu5_3)

		out = {
			'relu1_1': relu1_1,
			'relu1_2': relu1_2,

			'relu2_1': relu2_1,
			'relu2_2': relu2_2,

			'relu3_1': relu3_1,
			'relu3_2': relu3_2,
			'relu3_3': relu3_3,
			'relu3_4': relu3_4,

			'relu4_1': relu4_1,
			'relu4_2': relu4_2,
			'relu4_3': relu4_3,
			'relu4_4': relu4_4,

			'relu5_1': relu5_1,
			'relu5_2': relu5_2,
			'relu5_3': relu5_3,
			'relu5_4': relu5_4,
		}
		return out

class VGGLoss(nn.Module):
	def __init__(self):
		super(VGGLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = torch.nn.L1Loss()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		content_loss = 0.0
		# # content_loss += self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) * 0.1
		# # content_loss += self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) * 0.2
		content_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2']) * 1
		content_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2']) * 1
		content_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2']) * 2

		return content_loss / 4.

class SWDLoss(nn.Module):
	def __init__(self):
		super(SWDLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = SWD()
		# self.SWD = SWDLocal()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		N, C, H, W = x.shape  # 192*192
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		swd_loss = 0.0
		swd_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'], k=H//4//p) * 1  # H//4=48
		swd_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'], k=H//8//p) * 1  # H//4=24
		swd_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'], k=H//16//p) * 2  # H//4=12

		return swd_loss * 8 / 100.0

class SWD(nn.Module):
	def __init__(self):
		super(SWD, self).__init__()
		self.l1loss = torch.nn.L1Loss() 

	def forward(self, fake_samples, true_samples, k=0):
		N, C, H, W = true_samples.shape

		num_projections = C//2

		true_samples = true_samples.view(N, C, -1)
		fake_samples = fake_samples.view(N, C, -1)

		projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
		projections = torch.FloatTensor(projections).to(true_samples.device)
		projections = F.normalize(projections, p=2, dim=1)

		projected_true = projections @ true_samples
		projected_fake = projections @ fake_samples

		sorted_true, true_index = torch.sort(projected_true, dim=2)
		sorted_fake, fake_index = torch.sort(projected_fake, dim=2)
		return self.l1loss(sorted_true, sorted_fake).mean() 

class FilterLoss(nn.Module): # kernel_size%2=1
	def __init__(self):
		super(FilterLoss, self).__init__()

	def forward(self, filter_weight):  # [out, in, kernel_size, kernel_size]
		weight = filter_weight
		out_c, in_c, k, k = weight.shape 
		index = torch.arange(-(k//2), k//2+1, 1)

		index = index.to(filter_weight.device)
		index = index.unsqueeze(dim=0).unsqueeze(dim=0)  # [1, 1, kernel_size] 
		index_i = index.unsqueeze(dim=3)  # [1, 1, kernel_size, 1]  
		index_j = index.unsqueeze(dim=0)  # [1, 1, 1, kernel_size]  

		diff = torch.mean(weight*index_i, dim=2).abs() + torch.mean(weight*index_j, dim=3).abs()
		return diff.mean()
		
class MarginLoss(nn.Module):
	def __init__(self, opt, kl=False):
		super(MarginLoss, self).__init__()
		self.margin = 1.0  
		self.safe_radius = 4  # tea:3; stu:4
		self.scaling_steps = 2 
		self.temperature = 0.15 
		self.distill_weight = 15 
		self.perturb = opt.perturb
		self.kl = kl

	def forward(self, img1_1, img1_2, img2_1=None, img2_2=None, transformed_coordinates=None):
		device = img1_1.device
		loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
		pos_dist = 0.
		neg_dist = 0.
		distill_loss_all = 0.
		has_grad = False

		n_valid_samples = 0
		batch_size = img1_1.size(0)

		for idx_in_batch in range(batch_size):
			# Network output
			# shape: [c, h1, w1]
			dense_features1 = img1_1[idx_in_batch]
			c, h1, w1 = dense_features1.size()  # [256, 48, 48]

			# shape: [c, h2, w2]
			dense_features2 = img1_2[idx_in_batch]
			_, h2, w2 = dense_features2.size()  # [256, 48, 48]

			# shape: [c, h1 * w1]
			all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
			descriptors1 = all_descriptors1

			# Warp the positions from image 1 to image 2\
			# shape: [2, h1 * w1], coordinate in [h1, w1] dim,
			# dim 0: y, dim 1: x, positions in feature map
			fmap_pos1 = grid_positions(h1, w1, device)
			# shape: [2, h1 * w1], coordinate in image level (4 * h1, 4 * w1)
			# pos1 = upscale_positions(fmap_pos1, scaling_steps=self.scaling_steps)
			pos1 = fmap_pos1
			pos1, pos2, ids = warp(pos1, h1, w1, 
				transformed_coordinates[idx_in_batch], self.perturb)

			# print(descriptors1.shape, dense_features2.shape, transformed_coordinates.shape)
			# print(pos1.shape, pos2.shape, ids.shape)
			# print(pos1, '====', pos2, '====', ids)
			# exit()
			# shape: [2, num_ids]
			fmap_pos1 = fmap_pos1[:, ids]
			# shape: [c, num_ids]
			descriptors1 = descriptors1[:, ids]

			# Skip the pair if not enough GT correspondences are available
			if ids.size(0) < 128:
				continue

			# Descriptors at the corresponding positions
			# fmap_pos2 = torch.round(downscale_positions(pos2, \
			# 	scaling_steps=self.scaling_steps)).long()  # [2, hw]
			fmap_pos2 = torch.round(pos2).long()  # [2, hw]

			# [256, 48, 48] -> [256, hw]
			descriptors2 = F.normalize(
				dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)
			
			# [hw, 1, 256] @ [hw, 256, 1] -> [hw, hw]
			positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(1) @ \
				descriptors2.t().unsqueeze(2)).squeeze()  
				
			position_distance = torch.max(torch.abs(fmap_pos2.unsqueeze(2).float() - 
				fmap_pos2.unsqueeze(1)), dim=0)[0]  # [hw, hw]
			# print(position_distance)
			is_out_of_safe_radius = position_distance > self.safe_radius
			distance_matrix = 2 - 2 * (descriptors1.t() @ descriptors2)  # [hw, hw]
			negative_distance2 = torch.min(distance_matrix + (1 - 
				is_out_of_safe_radius.float()) * 10., dim=1)[0]  # [hw]

			all_fmap_pos1 = grid_positions(h1, w1, device)
			position_distance = torch.max(torch.abs(fmap_pos1.unsqueeze(2).float() - 
				all_fmap_pos1.unsqueeze(1)), dim=0)[0]
			# print(position_distance)
			is_out_of_safe_radius = position_distance > self.safe_radius
			distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
			negative_distance1 = torch.min(distance_matrix + (1 - 
				is_out_of_safe_radius.float()) * 10., dim=1)[0]

			# print(distance_matrix.shape, negative_distance1.shape)
			diff = positive_distance - torch.min(negative_distance1, negative_distance2)
			# diff = diff * 5.

			if not self.kl:
				loss = loss + torch.mean(F.relu(self.margin + diff))
			else:
				# distillation loss
				# student model correlation
				student_distance = torch.matmul(descriptors1.transpose(0, 1), descriptors2)
				student_distance = student_distance / self.temperature
				student_distance = F.log_softmax(student_distance, dim=1)

				# teacher model correlation
				teacher_dense_features1 = img2_1[idx_in_batch]
				c, h1, w1 = dense_features1.size()
				teacher_descriptors1 = F.normalize(teacher_dense_features1.view(c, -1), dim=0)
				teacher_descriptors1 = teacher_descriptors1[:, ids]

				teacher_dense_features2 = img2_2[idx_in_batch]
				teacher_descriptors2 = F.normalize(
					teacher_dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]], dim=0)
				
				teacher_distance = torch.matmul(
					teacher_descriptors1.transpose(0, 1), teacher_descriptors2)
				teacher_distance = teacher_distance / self.temperature
				teacher_distance = F.softmax(teacher_distance, dim=1)

				distill_loss = F.kl_div(student_distance, teacher_distance, \
					reduction='batchmean') * self.distill_weight
				distill_loss_all += distill_loss

				loss = loss + torch.mean(F.relu(self.margin + diff)) + distill_loss

			pos_dist = pos_dist + torch.mean(positive_distance)
			neg_dist = neg_dist + torch.mean(torch.min(negative_distance1, negative_distance2))

			has_grad = True
			n_valid_samples += 1
		
		if not has_grad:
			raise NotImplementedError

		loss = loss / n_valid_samples
		pos_dist = pos_dist / n_valid_samples
		neg_dist = neg_dist / n_valid_samples

		if not self.kl:
			return loss, pos_dist, neg_dist
		else:
			distill_loss_all = distill_loss_all / n_valid_samples
			return loss, pos_dist, neg_dist, distill_loss_all

