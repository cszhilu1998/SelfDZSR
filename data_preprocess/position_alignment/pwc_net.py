#-*- encoding: UTF-8 -*-

import torch
import sys
from functools import partial
import pickle

try:
	from pwc.correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python


# Borrow the code of the optical flow network (PWC-Net) from https://github.com/sniklaus/pytorch-pwc/
class PWCNET(torch.nn.Module):
	def __init__(self):
		super(PWCNET, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 
								81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 
							   81 + 128 + 2 + 2, 81, None ][intLevel + 0]
				
				self.backwarp_tenGrid = {}
				self.backwarp_tenPartial = {}

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, 
												  out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(
												  in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, 
												  kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96,
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, 
									kernel_size=3, stride=1, padding=1)
				)

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None
					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(
								tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)
					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(
								tenFirst=tenFirst, tenSecond=self.backwarp(tenInput=tenSecond, 
								tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}

			def backwarp(self, tenInput, tenFlow):
				index = str(tenFlow.shape) + str(tenInput.device)
				if index not in self.backwarp_tenGrid:
					tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
											tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
					tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
											tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
					self.backwarp_tenGrid[index] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

				if index not in self.backwarp_tenPartial:
					self.backwarp_tenPartial[index] = tenFlow.new_ones([ tenFlow.shape[0], 
															1, tenFlow.shape[2], tenFlow.shape[3] ])

				tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
									tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
				tenInput = torch.cat([ tenInput, self.backwarp_tenPartial[index] ], 1)

				tenOutput = torch.nn.functional.grid_sample(input=tenInput, 
							grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
							mode='bilinear', padding_mode='zeros', align_corners=False)

				tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

				return tenOutput[:, :-1, :, :] * tenMask

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()
				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 
									out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)

			def forward(self, tenInput):
				return self.netMain(tenInput)

		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner()

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight 
							   in torch.load('./pwc-net.pth').items() })

		# pickle.load = partial(pickle.load, encoding="latin1")
		# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

		# self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight 
		# 					   in torch.load('./pwc-net.pth', map_location=lambda storage, 
		# 					   loc: storage, pickle_module=pickle).items() })


	def forward(self, tenFirst, tenSecond):
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)
		objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
		objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
		objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
		objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
		objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)
		return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])




netNetwork = None

##########################################################

def estimate(tenFirst, tenSecond):
	global netNetwork

	if netNetwork is None:
		netNetwork = PWCNET().to(device).eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	#assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	tenPreprocessedFirst = tenFirst.to(device).view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.to(device).view(1, 3, intHeight, intWidth)

	intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
	intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

	tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
	tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

	tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
	tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

	return tenFlow[0, :, :, :].cpu()
# end

##########################################################


def backwarp_test(tenInput, tenFlow):
	tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
	tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

	backwarp_tenGrid = torch.cat([ tenHor, tenVer ], 1).to(device)
	# end

	backwarp_tenPartial = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
	tenInput = torch.cat([ tenInput, backwarp_tenPartial ], 1)

	tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

	tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

	return tenOutput[:, :-1, :, :] * tenMask, tenMask

if __name__ == '__main__':

	import numpy as np 
	import os
	import PIL
	import PIL.Image
	import sys
	from tqdm import tqdm
	import math
	import imageio

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Device: ', device)

	arguments_strFirst = '../short-focus_center.png'
	arguments_strSecond = '../telephoto.JPG'
	arguments_strOut = '../position_alignment/HR_warp.png'
	arguments_strmask = '../position_alignment/HR_mask.png'
	down_sampling_factor = 8

	tenFirst_init = torch.FloatTensor(np.ascontiguousarray(imageio.imread(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
	tenSecond_init = torch.FloatTensor(np.ascontiguousarray(imageio.imread(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))

	C, H, W = tenFirst_init.shape
	tenSecond_init = torch.nn.functional.interpolate(input=tenSecond_init.unsqueeze(0), size=(4*H,4*W), mode='bilinear', align_corners=True).squeeze(0)

	if tenFirst_init.shape[1] * 4 != tenSecond_init.shape[1]:
		tenSecond_init = tenSecond_init.transpose(2,1)
		print(tenFirst_init.shape, tenSecond_init.shape)
	
	C, H, W = tenSecond_init.shape
	h = (H//64) * 64 // down_sampling_factor
	w = (W//64) * 64 // down_sampling_factor

	tenFirst = torch.nn.functional.interpolate(input=tenFirst_init.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True)
	tenSecond = torch.nn.functional.interpolate(input=tenSecond_init.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=True)
	
	tenOutput = estimate(tenFirst.squeeze(0), tenSecond.squeeze(0)) 
	tenOutput = torch.nn.functional.interpolate(input=tenOutput.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True)

	tenOutput[:, 0:1, :, :] = tenOutput[:, 0:1, :, :] * (W/w)
	tenOutput[:, 1:2, :, :] = tenOutput[:, 1:2, :, :] * (H/h)

	out_first, tenMask = backwarp_test(tenSecond_init.unsqueeze(0).to(device), tenOutput.to(device))
	out_first = PIL.Image.fromarray(np.clip(out_first[0].detach().cpu().numpy()*255, 0, 255).astype(np.uint8).transpose(1, 2, 0)[..., ::-1])
	out_first.save(arguments_strOut)

	tenMask = PIL.Image.fromarray(np.clip(tenMask[0][0].detach().cpu().numpy()*255, 0, 255).astype(np.uint8))
	tenMask.save(arguments_strmask)