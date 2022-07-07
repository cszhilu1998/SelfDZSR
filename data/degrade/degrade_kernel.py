import cv2
import numpy as np
import random
import torch
from scipy import ndimage
from scipy.interpolate import interp2d
from .unprocess import unprocess, random_noise_levels, add_noise
from .process import process
from PIL import Image

def get_degrade_seq(sf):
    degrade_seq = []
    need_shift = False
    global_sf = None

    # -----------
    # down sample
    # -----------
    B_down = {
        "mode": "down",
        "sf": sf
    }
    mode = random.random()
    B_down["down_mode"] = "bilinear"


    degrade_seq.append(B_down)

    # --------------
    # gaussian noise
    # --------------
    B_noise = {
        "mode": "noise",
        "noise_level": random.randint(5, 30)  # 1, 19
    }
    degrade_seq.append(B_noise)

    # ----------
    # jpeg noise
    # ----------
    B_jpeg = {
        "mode": "jpeg",
        "qf": random.randint(60, 95)  # 40, 95
    }
    degrade_seq.append(B_jpeg)

    # # -------------------
    # # Processed camera sensor noise
    # # -------------------
    # B_camera = {
    #     "mode": "camera",
    # }
    # degrade_seq.append(B_camera)

    # -------
    # shuffle
    # -------
    random.shuffle(degrade_seq)

    return degrade_seq


def degrade_kernel(img, sf=4):
    h, w, c = np.array(img).shape
    degrade_seq = get_degrade_seq(sf)
    for degrade_dict in degrade_seq:
        mode = degrade_dict["mode"]
        if mode == "blur":
            img = get_blur(img, degrade_dict)
        elif mode == "down":
            img = get_down(img, degrade_dict)
        elif mode == "noise":
            img = get_noise(img, degrade_dict)
        elif mode == 'jpeg':
            img = get_jpeg(img, degrade_dict)
        elif mode == 'camera':
            img = get_camera(img, h, w, degrade_dict)
        elif mode == 'restore':
            img = get_restore(img, w, h, degrade_dict)
        else:
            exit(mode)
        # print(mode, np.array(img).shape)
    # print_degrade_seg(degrade_seq)
    return img, degrade_seq


def get_blur(img, degrade_dict):

    img = np.array(img)
    k_size = degrade_dict["kernel_size"]
    if degrade_dict["is_aniso"]:
        sigma_x = degrade_dict["x_sigma"]
        sigma_y = degrade_dict["y_sigma"]
        angle = degrade_dict["rotation"]
    else:
        sigma_x = degrade_dict["sigma"]
        sigma_y = degrade_dict["sigma"]
        angle = 0

    kernel = np.zeros((k_size, k_size))
    d = k_size // 2
    for x in range(-d, d+1):
        for y in range(-d, d+1):
            kernel[x+d][y+d] = get_kernel_pixel(x, y, sigma_x, sigma_y)
    M = cv2.getRotationMatrix2D((k_size//2, k_size//2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
    kernel = kernel / np.sum(kernel)
    img = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='reflect')

    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))


def get_down(img, degrade_dict):
    img = np.array(img)
    sf = degrade_dict["sf"]
    mode = degrade_dict["down_mode"]
    h, w, c = img.shape
    if mode == "nearest":
        img = img[0::sf, 0::sf, :]
    elif mode == "bilinear":
        new_h, new_w = int(h/sf), int(w/sf)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    elif mode == "bicubic":
        new_h, new_w = int(h/sf), int(w/sf)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))


def get_noise(img, degrade_dict):
    noise_level = degrade_dict["noise_level"]
    img = np.array(img)
    img = img + np.random.normal(0, noise_level, img.shape)
    return Image.fromarray(np.uint8(np.clip(img, 0.0, 255.0)))


def get_jpeg(img, degrade_dict):
    qf = degrade_dict["qf"]
    img = np.array(img)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),qf] # (0,100),higher is better,default is 95
    _, encA = cv2.imencode('.jpg',img,encode_param)
    Img = cv2.imdecode(encA,1)
    return Image.fromarray(np.uint8(np.clip(Img, 0.0, 255.0)))


def get_camera(img, h, w, degrade_dict):
    img = torch.from_numpy(np.array(img)) / 255.0
    deg_img, features = unprocess(img)
    shot_noise, read_noise = random_noise_levels()
    deg_img = add_noise(deg_img, shot_noise, read_noise)
    deg_img = deg_img.unsqueeze(0)
    features['red_gain'] = features['red_gain'].unsqueeze(0)
    features['blue_gain'] = features['blue_gain'].unsqueeze(0)
    features['cam2rgb'] = features['cam2rgb'].unsqueeze(0)
    deg_img = process(deg_img, features['red_gain'], features['blue_gain'], features['cam2rgb'])
    deg_img = deg_img.squeeze(0)
    deg_img = torch.clamp(deg_img * 255.0, 0.0, 255.0).numpy()
    deg_img = deg_img.astype(np.uint8)
    return Image.fromarray(deg_img)


def get_restore(img, h, w, degrade_dict):
    need_shift = degrade_dict["need_shift"]
    sf = degrade_dict["sf"]
    img = np.array(img)
    mode = degrade_dict["up_mode"]
    if mode == "bilinear":
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
    if need_shift:
        img = shift_pixel(img, int(sf))
    return Image.fromarray(img)

def get_kernel_pixel(x, y, sigma_x, sigma_y):
    return 1/(2*np.pi*sigma_x*sigma_y)*np.exp(-((x*x/(2*sigma_x*sigma_x))+(y*y/(2*sigma_y*sigma_y))))


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def print_degrade_seg(degrade_seq):
    for degrade_dict in degrade_seq:
        print(degrade_dict)
