# SelfDZSR (ECCV 2022)

Official PyTorch implementation of [**SelfDZSR**](https://arxiv.org/abs/2203.01325) 


> [**Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations**](https://arxiv.org/abs/2203.01325) <br>
> ECCV, 2022 <br>
> [Zhilu Zhang](https://scholar.google.com/citations?user=8pIq2N0AAAAJ), [Ruohao Wang](https://scholar.google.com/citations?user=o1FPNwQAAAAJ), [Hongzhi Zhang](https://scholar.google.com/citations?user=Ysk4WBwAAAAJ), [Yunjin Chen](https://scholar.google.com/citations?user=CaoMb2AAAAAJ), [Wangmeng Zuo](https://scholar.google.com/citations?user=rUOpCEYAAAAJ)
<br>Harbin Institute of Technology, China

The extended version of SelfDZSR has been accepted by IEEE TPAMI in 2024.


> [**Self-Supervised Learning for Real-World Super-Resolution from Dual and Multiple Zoomed Observations**](https://arxiv.org/abs/2405.02171) <br>
> IEEE TPAMI, 2024 <br>
> [Zhilu Zhang](https://scholar.google.com/citations?user=8pIq2N0AAAAJ), [Ruohao Wang](https://scholar.google.com/citations?user=o1FPNwQAAAAJ), [Hongzhi Zhang](https://scholar.google.com/citations?user=Ysk4WBwAAAAJ), [Wangmeng Zuo](https://scholar.google.com/citations?user=rUOpCEYAAAAJ)
<br>Harbin Institute of Technology, China
<br>GitHub: https://github.com/cszhilu1998/SelfDZSR_PlusPlus


## 1. Framework

<p align="center"><img src="introduction.png" width="95%"></p>
<p align="center">Overall pipeline of proposed SelfDZSR in the training and testing phase.</p>

- In the training, the center part of the short-focus and telephoto image is cropped respectively as the input LR and Ref, and the whole telephoto image is taken as the GT. The auxiliary-LR is generated to guide the alignment of LR and Ref towards GT.

- In the testing, SelfDZSR can be directly deployed to super-solve the whole short-focus image with the reference of the telephoto image.

## 2. Preparation and Datasets

- **Prerequisites**
    - Python 3.x and PyTorch 1.6.
    - OpenCV, NumPy, Pillow, tqdm, lpips, scikit-image and tensorboardX.

- **Dataset**
    - **Nikon camera images** and **CameraFusion dataset** can be downloaded from this [link](https://drive.google.com/drive/folders/1XTxU6iPxs_MZTM8g_hUsq0REWqB0P-AB?usp=sharing).
   
   
- **Data pre-processing**
    - If you want to pre-process additional short-focus images and telephoto images, we provide a demo in [`./data_preprocess`](./data_preprocess). (2022/9/13)
    

## 3. Quick Start

### 3.1 Pre-trained models

- For simplifying the training process, we provide the pre-trained models of feature extractors and auxiliary-LR generator. The models for Nikon camera images and CameraFusion dataset are put in the `./ckpt/nikon_pretrain_models/` and `./ckpt/camerafusion_pretrain_models/` folder, respectively.

- For direct testing, we provide the four pre-trained DZSR models (`nikon_l1`, `nikon_l1sw`, `camerafusion_l1` and `camerafusion_l1sw`) in the `./ckpt/` folder. Taking `nikon_l1sw` as an example, it represents the model trained on the Nikon camera images using $l_1$ and sliced Wasserstein (SW) loss terms.


### 3.2 Training

- For Nikon camera images, modify `dataroot` in `train_nikon.sh` and then run:

    [`sh train_nikon.sh`](train_nikon.sh)

- For CameraFusion dataset, modify `dataroot` in `train_camerafusion.sh` and then run:
    
    [`sh train_camerafusion.sh`](train_camerafusion.sh)

### 3.3 Testing

- For Nikon camera images, modify `dataroot` in `test_nikon.sh` and then run:

    [`sh test_nikon.sh`](test_nikon.sh)

- For CameraFusion dataset, modify `dataroot` in `test_camerafusion.sh` and then run:
    
    [`sh test_camerafusion.sh`](test_camerafusion.sh)

### 3.4 Note

- You can specify which GPU to use by `--gpu_ids`, e.g., `--gpu_ids 0,1`, `--gpu_ids 3`, `--gpu_ids -1` (for CPU mode). In the default setting, all GPUs are used.
- You can refer to [options](./options/base_options.py) for more arguments.


## 4. Citation
If you find it useful in your research, please consider citing:

    @inproceedings{SelfDZSR,
        title={Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations},
        author={Zhang, Zhilu and Wang, Ruohao and Zhang, Hongzhi and Chen, Yunjin and Zuo, Wangmeng},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2022}
    }

    @article{SelfDZSR_PlusPlus,
        title={Self-Supervised Learning for Real-World Super-Resolution from Dual and Multiple Zoomed Observations},
        author={Zhang, Zhilu and Wang, Ruohao and Zhang, Hongzhi and Zuo, Wangmeng},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
        year={2024},
        publisher={IEEE}
    }

## 5. Acknowledgement

This repo is built upon the framework of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and we borrow some code from [C2-Matching](https://github.com/yumingj/C2-Matching) and [DCSR](https://github.com/Tengfei-Wang/DCSR), thanks for their excellent work!