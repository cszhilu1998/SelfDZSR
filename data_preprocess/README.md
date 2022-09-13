# Data Pre-processing for SelfDZSR

A demo for data pre-processing of short-focus image and telephoto image.

## 1. Crop short-focus image

- Given the short-focus image `./short-focus.JPG` and telephoto image `./telephoto.JPG`, we first crop the short-focus image based on the focal-length ratio between the two images.

    [`python crop_center.py`](./crop_center/crop_center.py)

- Then we can get `./short-focus_center.png`, it has roughly the same scenes as the telephoto image.


## 2. Position alignment

- We use PWC-Net to align `./short-focus_center.png` and `./telephoto.JPG` spatially.

    [`python pwc_net.py`](./position_alignment/pwc_net.py)

- Then we can get the aligned telephoto image, named `./position_alignment/HR_warp.png`.


## 3. Color alignment

- Color alignment is performed between `./short-focus_center.png` and  `./position_alignment/HR_warp.png`.

    [`python color_alignment.py`](./color_alignment/color_alignment.py)

- Finally, the low-resolution image `./color_alignment/LR.png`, high-resolution image `./color_alignment/HR.png`, and reference image `./color_alignment/Ref.png` can be obtained.
