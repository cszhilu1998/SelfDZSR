import numpy as np
import glob
import os
import cv2


sr_scale = 4

def cut_out_mask(lr, hr, mask, p=0):
	hr_p = sr_scale * p
	return lr[p:-p, p:-p], \
		   hr[hr_p:-hr_p, hr_p:-hr_p],\
		   mask[hr_p:-hr_p, hr_p:-hr_p]
			
def color_alignment(LR_IMG, HR_IMG):
    LR = LR_IMG
    HR = HR_IMG
    H, W, _ = HR.shape
    LR = cv2.resize(LR, (W, H), interpolation=cv2.INTER_CUBIC)

    gray_thres = 0.05
    LR = cv2.cvtColor(LR, cv2.COLOR_BGR2YCrCb)
    HR = cv2.cvtColor(HR, cv2.COLOR_BGR2YCrCb)
    LR_Y = LR[..., 0]
    HR_Y = HR[..., 0]
    LR_YY= LR_Y.astype(np.float32) /255.0
    HR_YY = HR_Y.astype(np.float32) /255.0

    error_ = np.abs(LR_Y  - HR_Y)
    error_s = error_.flatten()
    thres_ = np.sort(error_s)[::-1]
    thres_ = thres_[int(thres_.shape[0] * gray_thres)]
    error_index = error_ < thres_
    LR_Y = LR_YY[error_index]
    HR_Y = HR_YY[error_index]

    A = np.array([HR_Y, np.ones_like(HR_Y)]).transpose(1, 0)
    B = LR_Y

    x = np.linalg.pinv(np.conjugate(A).T @ A) @ (np.conjugate(A).T @ B)
    HR[..., 0] = ((HR_YY * x[0] + x[1]) * 255).round().clip(0, 255).astype(np.uint8)
    HR = cv2.cvtColor(HR, cv2.COLOR_YCrCb2BGR)
    LR = cv2.cvtColor(LR, cv2.COLOR_YCrCb2BGR)
    ## WB
    HR = HR.astype(np.float32)
    LR = LR.astype(np.float32)

    mean1 = LR.mean()
    mean2 = HR.mean()
    for i in range(3):
        ave_ch1 = LR[...,i].mean()
        ave_ch2 = HR[...,i].mean()
        HR[..., i] = HR[..., i] * (mean2 * ave_ch1) / (mean1 * ave_ch2)

    HR = HR.round().clip(0, 255).astype(np.uint8)
    LR = LR.astype(np.uint8)
    return HR

def crop_center(hr, fw=0.25, fh=0.25):
	ih, iw = hr.shape[0:2]
	lr_patch_h, lr_patch_w = round(ih * fh), round(iw * fw)
	ph = ih // 2 - lr_patch_h // 2
	pw = iw // 2 - lr_patch_w // 2

	return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

if __name__ == '__main__':

    lr_file = '../short-focus_center.png'
    hr_file = '../position_alignment/HR_warp.png'
    hr_mask_file = '../position_alignment/HR_mask.png'

    vislr = cv2.imread(lr_file, -1)
    vishr = cv2.imread(hr_file, -1)
    vismask = cv2.imread(hr_mask_file, -1)

    cut_lr, cut_hr, cut_mask = cut_out_mask(vislr, vishr, vismask, p=80)

    if (cut_mask-255).mean() != 0:
        print(hr_file, ' needs to be removed more boundaries.')
        exit()
    
    cut_hr_color = color_alignment(cut_lr, cut_hr)

    cv2.imwrite('./LR.png', cut_lr)
    cv2.imwrite('./HR.png', cut_hr_color)

    ref = crop_center(cut_hr_color, fw=1./sr_scale, fh=1./sr_scale)
    cv2.imwrite('./Ref.png', ref)
