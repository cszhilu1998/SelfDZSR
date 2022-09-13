import cv2
from PIL import Image


FOCAL_CODE = 37386

def readFocal_pil(image_path):
    if 'ARW' in image_path:
        image_path = image_path.replace('ARW','JPG')

    img = Image.open(image_path)
    exif_data = img._getexif()

    return float(exif_data[FOCAL_CODE]) # [0]/exif_data[FOCAL_CODE][1]

def crop_center(hr, fw=0.25, fh=0.25):
	ih, iw = hr.shape[0:2]
	lr_patch_h, lr_patch_w = round(ih * fh), round(iw * fw)
	ph = ih // 2 - lr_patch_h // 2
	pw = iw // 2 - lr_patch_w // 2

	return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

if __name__ == '__main__':
	hr_path = '../telephoto.JPG'
	lr_path = '../short-focus.JPG'
	hr_f = readFocal_pil(hr_path)
	lr_f = readFocal_pil(lr_path)

	vislr = cv2.imread(lr_path)
	vis_center = crop_center(vislr, fw=lr_f/hr_f, fh=lr_f/hr_f)
	cv2.imwrite('../short-focus_center.png', vis_center)


