## PaCa-ViT (CVPR'23) <br> <sub>Official PyTorch Implementation</sub>

This repo contains the PyTorch version of model definitions (*Tiny, Small, Base*), training code and pre-trained weights for ImageNet-1k classification, MS-COCO object detection and instance segmentation, and MIT-ADE20k image semantic segmenation for our PaCa-ViT paper. It is refactored using PyTorch 2.0 and the latest [timm](https://github.com/huggingface/pytorch-image-models), [mmdetection 3.x](https://github.com/open-mmlab/mmdetection/tree/v3.0.0) and [mmsegmentation 1.x](https://github.com/open-mmlab/mmsegmentation/tree/v1.0.0). The trained checkpoints are converted accordingly. We thank the teams of those open-sourced packages.  

> [**PaCa-ViT: Learning Patch-to-Cluster Attention in Vision Transformers**](https://arxiv.org/abs/2203.11987)<br>
> Ryan Grainger, Thomas Paniagua, Xi Song, Naresh Cuntoor, Mun Wai Lee and Tianfu Wu\
> <br>NC State University, BlueHalo, X. Song is an independent researcher.<br>

Vision Transformers (ViTs) are built on the assumption of treating image patches as ``visual tokens" and learn patch-to-patch attention. The patch embedding based tokenizer has a semantic gap with respect to its counterpart, the textual tokenizer. The patch-to-patch attention suffers from the quadratic complexity issue, and also makes it non-trivial to explain learned ViTs. To address these issues in ViT, this paper proposes to learn **Patch-to-Cluster attention (PaCa)** in ViTs. 

<p align="center">
<img src="assets/paca_teaser.png" width="90%" height="70%" class="center">
</p>

Queries in our PaCa-ViT starts with patches, while keys and values are directly based on clustering (with a predefined small number of clusters). The clusters are learned end-to-end, leading to better tokenizers and inducing joint clustering-for-attention and attention-for-clustering for better and interpretable models. The quadratic complexity is relaxed to linear complexity. The proposed PaCa module is used in designing efficient and interpretable ViT backbones and semantic segmentation head networks. 

<p align="center">
<img src="assets/paca_scheme.png" width="90%" height="50%" class="center">
</p>

We study four aspects of the PaCa module: 

> **Where to compute the cluster assignments?** Consider the stage-wise pyramidical architecture of assembling ViT blocks, a stage consists of a number of blocks. We test two settings: *block-wise* by computing the cluster assignment for each block, or *stage-wise* by computing it only in the first block in a stage and then sharing it with the remaining blocks. Both give comparable performance. The latter is more efficient when the model becomes deeper. 

> **How to compute the cluster assignment?** We also test two settings: using 2D convolution or Multi-Layer Perceptron (MLP) based implementation. Both have similar performance. The latter is more generic and sheds light on exploiting PaCa for more general *Token-to-Cluster attention (ToCa)* in a domain agnostic way. 

<p align="center">
<img src="assets/paca-vit-onsite.png" width="90%" height="50%" class="center">
</p>

> **How to leverage an external clustering teacher?** We investigate a method of exploiting a lightweight convolution neural network  in learning the cluster assignments that are shared by all blocks in a stage. It gives some interesting observations, and potentially pave a way for distilling large foundation models. 

<p align="center">
<img src="assets/paca-vit-teacher.png" width="90%" height="50%" class="center">
</p>

> **What if the number of clusters is known?** We further extend the PaCa module in designing an effective head sub-network for dense prediction tasks such as image semantic segmentation where the number of clusters $M$ is available based on the ground-truth number of classes and the learned cluster assignment $\mathcal{C}_{N, M}$ has direct supervision. The PaCa segmentation head significantly improves the performance with reduced model complexity. 

<p align="center">
<img src="assets/paca-seghead.png" width="90%" height="50%" class="center">
</p>

## Results and Trained Models
### ImageNet-1K (224x224) trained weights 

> *(please refer to the paper for details of the settings)*

| name | acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:| :---:|:---:|
| PaCa-Tiny (conv) |  80.9 | 12.2M  | 3.2G | [model]() |
| PaCa-Small (conv) |  83.08 | 22.0M  | 5.5G | [model]() |
| PaCa-Small (mlp) |  83.13 | 22.6M  | 5.9G | [model]() |
| PaCa-Small (teacher) |  83.17 | 21.1M  | 5.4G | [model]() |
| PaCa-Base (conv)|  83.96 | 46.9M  | 9.5G | [model]() |
| PaCa-Base (teacher)|  84.22 | 46.7M  | 9.7G | [model]() |

## Installation
We provide self-contained [installation scripts](install.sh) and [environment configurations](environment.yaml). 

```shell
git clone https://github.com/iVMCL/PaCaViT.git 
cd PaCaViT 
ln -s ./models ./classification/
ln -s ./models ./detection/
ln -s ./models ./segmentation/
chmod +x ./*.sh
./install.sh pacavit 
```

## Training
We provide scripts for training models using a single-node GPU server (e.g., 8 NVIDIA GPUs).
### **ImageNet-1k Classification**
We borrow from the [timm](https://github.com/huggingface/pytorch-image-models) package. 

> *Data preparation.* Download the [ImageNet dataset](http://image-net.org/download) to `YOUR_IMNET_PATH` and unzip it. Move validation images to labeled subfolders using this [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). 
```shell
cd $YOUR_IMNET_PATH/Data/CLS-LOC/val
chmod +x ./valprep.sh
./valprep.sh
```
> Add the symbolic link to the ImageNet dataset, 
```shell
cd PaCaViT
mkdir datasets
ln -s $YOUR_IMNET_PATH/Data/CLS-LOC ./datasets/IMNET
```
> Use the provided self-contained [training script](./classification/train_timm.sh) and [model configurations](./classification/configs/imagenet_vit_adamw.yml) to train a model, 
```
train_timm.sh \
Config_file \
Model_name \
Dataset_name \
Img_size \
Remove_old_if_exist_0_or_1 \
Resume_or_not_if_exist \
Exp_name \
Tag \
Gpus \
Nb_gpus \
Workers \
Port \
[others]
```    
e.g., train a PaCa-Tiny model [pacavit_tiny_p2cconv_100_0](./models/paca_vit.py#L935), 
```shell 
cd PaCaViT/classification 
./train_timm.sh configs/imagenet_vit_adamw.yml pacavit_tiny_p2cconv_100_0 IMNET 224 1 1 cvpr23 try1 0,1,2,3,4,5,6,7 8 8 23900
```
The training results will be saved in `PaCaViT/work_dirs/classification/cvpr23/pacavit_tiny_p2cconv_100_0_try1/` before the training is completed. After completed, they will be auto-moved to 
`PaCaViT/work_dirs/classification/cvpr23/TrainingFinished/pacavit_tiny_p2cconv_100_0_try1/`

### **MS-COCO Object Detection and Instance Segmentation**
We borrow from the [mmdetection 3.x](https://github.com/open-mmlab/mmdetection/tree/v3.0.0) package. 

> *Data preparation*: Download [COCO 2017 datasets](https://cocodataset.org/#download) to `YOUR_COCO_PATH`. 

> Add the symbolic link to the ImageNet dataset, 
```shell
cd PaCaViT/datasets
ln -s $YOUR_COCO_PATH ./datasets/coco
```

> Use the provided [training script](./detection/train_mmdet.sh) and select one of [the configuration files](./detection/configs/paca_vit/mask_rcnn_1x/). 
```shell
cd PaCaViT/detection
chmod +x ./*.sh
```

```
train_mmdet.sh \
Relative_config_filename \
Remove_old_if_exist_0_or_1 \
Exp_name \
Tag \
gpus \
nb_gpus \
port \
[others]
```
e.g., train a Mask R-CNN with the PaCa-Tiny backbone [pacavit_tiny_p2cconv_100_0_downstream](./models/paca_vit.py#L1477), 
```shell 
cd PaCaViT/detection 
./train_mmdet.sh configs/paca_vit/mask_rcnn_1x/mask_rcnn_pacavit_tiny_p2cconv_100_0_mstrain_480_800_1x_coco.py 1 cvpr23 try1 0,1,2,3,4,5,6,7 8 23900
```
The training results will be saved in `PaCaViT/work_dirs/detection/cvpr23/mask_rcnn_pacavit_tiny_p2cconv_100_0_mstrain_480_800_1x_coco_try1/`

### **MIT-ADE20k Image Semantic Segmentation**
We borrow from the [mmsegmentation 1.x](https://github.com/open-mmlab/mmsegmentation/tree/v1.0.0) package. 

> *Data preparation*: Download [MIT ADE2Ok 2016 dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) to `YOUR_ADE_PATH`. 

> Add the symbolic link to the ImageNet dataset, 
```shell
cd PaCaViT/datasets
ln -s $YOUR_ADE_PATH ./datasets/ADEChallengeData2016
```

> Use the provided [training script](./segmentation/train_mmseg.sh) and select one of [the configuration files](./segmentation/configs/paca_vit/). 
```shell
cd PaCaViT/segmentation
chmod +x ./*.sh
```

```
train_mmseg.sh \
Relative_config_filename \
Remove_old_if_exist_0_or_1 \
Exp_name \
Tag \
gpus \
nb_gpus \
port \
[others]
```
e.g., train the segmentor with the [PaCa segmentation head](./segmentation/mmseg_custom/models/decode_heads/paca_head.py) and the PaCa-Tiny backbone [pacavit_tiny_p2cconv_100_0_downstream](./models/paca_vit.py#L1477), 
```shell 
cd PaCaViT/segmentation 
./train_mmseg.sh configs/paca_vit/paca_head/pacahead_pacavit_tiny_p2cconv_100_0_512x512_160k_ade20k.py 1 cvpr23 try1 0,1,2,3,4,5,6,7 8 23900
```
The training results will be saved in `PaCaViT/work_dirs/segmentation/cvpr23/pacahead_pacavit_tiny_p2cconv_100_0_512x512_160k_ade20k_try1/`

## Evaluation
### Accuracy 
Please refer to the provided self-contained evaluation scripts for [ImageNet-1k evaluation](./classification/validate.sh), [MS-COCO evaluation](./detection/test_mmdet.sh) and [MIT-ADE20k evaluation](./segmentation/test_mmseg.sh). 

### Model Parameters and FLOPs
Please refer to the provided scripts for [benchmarking a feature backbone](./classification/benchmark.sh), [benchmarking a detector](./detection/get_flops.py) and [benchmarking a segmentator](./segmentation/get_flops.py). 


## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@inproceedings{Grainger2023PaCaViT,
  title={PaCa-ViT: Learning Patch-to-Cluster Attention in Vision Transformers},
  author={Ryan Grainger and Thomas Paniagua and Xi Song and Naresh Cuntoor and Mun Wai Lee and Tianfu Wu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
