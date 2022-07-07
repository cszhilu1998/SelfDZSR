"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import time
from functools import wraps
import torch
import random
import cv2
import torch
import colour_demosaicing
import glob


def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(1)
        return ret
    return wrapper

@loop_until_success
def loop_print(*args, **kwargs):
    print(*args, **kwargs)

@loop_until_success
def torch_save(*args, **kwargs):
    torch.save(*args, **kwargs)

def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(0, h, device=device).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(0, w, device=device).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)

def warp(pos1, max_h, max_w, transformed_coordinates, perturb):
    np.random.seed(0)
    device = pos1.device  # pos1 shape: (2, 48*48)
    ids = torch.arange(0, pos1.size(1), device=device)  # [0, 48*48]

    # transformed_coordinates = transformed_coordinates[::4, ::4, :2]
    transformed_coordinates = transformed_coordinates[::1, ::1, :2]
    # dim 0: x, dim 1: y
    pos2 = transformed_coordinates.permute(2, 0, 1).reshape(2, -1)
    transformed_x = pos2[0, :]
    transformed_y = pos2[1, :]

    # eliminate the outlier pixels
    valid_ids_x = torch.min(transformed_x > perturb, 
        transformed_x < (max_w - perturb))
    valid_ids_y = torch.min(transformed_y > perturb, 
        transformed_y < (max_h - perturb))

    valid_ids = torch.min(valid_ids_x, valid_ids_y)

    ids = ids[valid_ids]
    pos1 = pos1[:, valid_ids]
    pos2 = pos2[:, valid_ids]

    pos2 = pos2[[1, 0], :]
    # print(ids.shape, transformed_x, transformed_y)
    return pos1, pos2, ids

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images
	
def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def image_pair_generation(img,
                          random_perturb_range=(0, 32),
                          cropping_window_size=160):

    if img is not None:
        shape1 = img.shape
        h = shape1[0]
        w = shape1[1]
    else:
        h = 160
        w = 160
    # print(cropping_window_size, h, w)
    # ===== in image-1
    cropS = cropping_window_size

    x_topleft = np.random.randint(random_perturb_range[1], 
                high=max(w, w - cropS - random_perturb_range[1]))
    y_topleft = np.random.randint(random_perturb_range[1], 
                high=max(h, h - cropS - random_perturb_range[1]))

    x_topright = x_topleft + cropS
    y_topright = y_topleft

    x_bottomleft = x_topleft
    y_bottomleft = y_topleft + cropS

    x_bottomright = x_topleft + cropS
    y_bottomright = y_topleft + cropS

    tl = (x_topleft, y_topleft)
    tr = (x_topright, y_topright)
    br = (x_bottomright, y_bottomright)
    bl = (x_bottomleft, y_bottomleft)

    rect1 = np.array([tl, tr, br, bl], dtype=np.float32)

    # ===== in image-2
    x2_topleft = x_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topleft = y_topleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_topright = x_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_topright = y_topright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomleft = x_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomleft = y_bottomleft + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    x2_bottomright = x_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])
    y2_bottomright = y_bottomright + np.random.randint(
        random_perturb_range[0], random_perturb_range[1]) * np.random.choice(
            [-1.0, 1.0])

    tl2 = (x2_topleft, y2_topleft)
    tr2 = (x2_topright, y2_topright)
    br2 = (x2_bottomright, y2_bottomright)
    bl2 = (x2_bottomleft, y2_bottomleft)

    rect2 = np.array([tl2, tr2, br2, bl2], dtype=np.float32)

    # ===== homography
    H = cv2.getPerspectiveTransform(src=rect1, dst=rect2)

    try:
        H_inverse = np.linalg.inv(H)
    except:
        print('Singular matrix. Try again.')
        img_warped, H, H_inverse = \
            image_pair_generation(img, random_perturb_range, cropping_window_size)
        return img_warped, H, H_inverse

    # if img is not None:
    img_warped = cv2.warpPerspective(src=img, M=H_inverse, dsize=(w, h))
    return img_warped, H, H_inverse
    # else:
    #     return H_inverse

def augment_func(img, hflip, vflip, rot90):  # CxHxW
    if hflip:   img = img[:, :, ::-1]
    if vflip:   img = img[:, ::-1, :]
    if rot90:   img = img.transpose(0, 2, 1)
    return np.ascontiguousarray(img)

def augment(*imgs):  # CxHxW
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    return (augment_func(img, hflip, vflip, rot90) for img in imgs)

def calc_psnr_np(sr, hr, range=255.):
    """ calculate psnr by numpy

    Params:
    sr : numpy.uint8
        super-resolved image
    hr : numpy.uint8
        high-resolution ground truth
    scale : int
        super-resolution scale
    """
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / range
    # valid = diff[:, :, shave:-shave, shave:-shave]
    mse = np.power(diff, 2).mean()
    return -10 * math.log10(mse)

def calc_psnr(sr, hr, range=255.):
    """ calculate psnr by torch

    Params:
    sr : torch.float32
        super-resolved image
    hr : torch.float32
        high-resolution ground truth
    scale : int
        super-resolution scale
    """
    # shave = 2
    with torch.no_grad():
        diff = (sr - hr) / range
        # diff = diff[:, :, shave:-shave, shave:-shave]
        mse = torch.pow(diff, 2).mean()
        # print(mse)
        # print((-10 * torch.log10(mse)).item())
        return (-10 * torch.log10(mse)).item()

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def print_numpy(x, val=True, shp=True):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def prompt(s, width=66):
    print('='*(width+4))
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print('= ' + s.center(width) + ' =')
    else:
        for s in ss:
            for i in split_str(s, width):
                print('= ' + i.ljust(width) + ' =')
    print('='*(width+4))

def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss
