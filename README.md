# Object DGCNN & DETR3D

This repo contains the implementations of Object DGCNN (https://arxiv.org/abs/2110.06923) and DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.  

### Prerequisite

1. mmcv (https://github.com/open-mmlab/mmcv)

2. mmdet (https://github.com/open-mmlab/mmdetection)

3. mmseg (https://github.com/open-mmlab/mmsegmentation)

4. mmdet3d (https://github.com/open-mmlab/mmdetection3d)

### Data
1. Follow the mmdet3d to process the data.

### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/ 

2. For example, to train Object-DGCNN with pillar on 8 GPUs, please use

`tools/dist_train.sh projects/configs/obj_dgcnn/pillar.py 8`

### Evaluation using pretrained models
1. Download the weights accordingly.  

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[DETR3D, ResNet101 w/ DCN](./projects/configs/detr3d/detr3d_res101_gridmask.py)|34.7|42.2|[model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing)|
|[above, + CBGS](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)|34.9|43.4|[model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing)|
|[DETR3D, VoVNet on trainval, evaluation on test set](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py)| 41.2 | 47.9 |[model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing)|

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[Object DGCNN, pillar](./projects/configs/obj_dgcnn/pillar.py)|53.2|62.8|[model](https://drive.google.com/file/d/1nd6-PPgdb2b2Bi3W8XPsXPIo2aXn5SO8/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1A98dWp7SBOdMpo1fHtirwfARvpE38KOn/view?usp=sharing)|
|[Object DGCNN, voxel](./projects/configs/obj_dgcnn/voxel.py)|58.6|66.0|[model](https://drive.google.com/file/d/1zwUue39W0cAP6lrPxC1Dbq_gqWoSiJUX/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1pjRMW2ffYdtL_vOYGFcyg4xJImbT7M2p/view?usp=sharing)|


2. To test, use  
`tools/dist_test.sh projects/configs/obj_dgcnn/pillar_cosine.py /path/to/ckpt 8 --eval=bbox`

 
If you find this repo useful for your research, please consider citing the papers

```
@inproceedings{
   obj-dgcnn,
   title={Object DGCNN: 3D Object Detection using Dynamic Graphs},
   author={Wang, Yue and Solomon, Justin M.},
   booktitle={2021 Conference on Neural Information Processing Systems ({NeurIPS})},
   year={2021}
}
```

```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```
