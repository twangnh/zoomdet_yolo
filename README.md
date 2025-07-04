

<div align="center">   

# Adaptive Image Zoom-in with Bounding Box Transformation for Aerial Object Detection
</div>

This codebase implements the paper *"Adaptive Image Zoom-in with Bounding Box Transformation for Aerial Object Detection"* submitted 
to **ISPRS Journal of Photogrammetry and Remote Sensing**

YOLOv8 is used as the base object detection architecture in this repository.


## Installation

Please refer to [mmyolo](https://github.com/open-mmlab/mmyolo) for installation instructions. 


## Data Preparation
Download the datasets from official released sources:
[SeaDroneSee](https://seadronessee.cs.uni-tuebingen.de/dataset)
[VisDrone](https://aiskyeye.com/home/)
[UAVDT](https://datasetninja.com/uavdt)
,and convert the annotations to COCO json format.

prepare the data folder as:

```
${ROOT}
|-- data
|   |-- VisDrone
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- instances_train.json
|   |   |-- val
|   |   |   |-- images
|   |   |   |-- instances_train.json
|   |-- UAVDT
|   |-- ...
|   |-- SeaDroneSee
|   |-- ...
```

## Usage

### Train with VisDrone

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1001 ./tools/train.py ./configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_visdrone.py --launcher pytorch
```
### Test with VisDrone

```

python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py ./configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_visdrone.py work_dirs/yolov8_s_syncbn_fast_8xb16-500e_visdrone_saliency/best_coco_bbox_mAP_epoch_30.pth```
```
> replace the checkpoint with the corresponding trained one on the dataset


### Train with UAVDT

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1001 ./tools/train.py ./configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_UAVDT.py --launcher pytorch
```
### Test with UAVDT

```

python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py ./configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_UAVDT.py work_dirs/yolov8_s_syncbn_fast_8xb16-500e_visdrone_saliency/best_coco_bbox_mAP_epoch_30.pth```
```
> replace the checkpoint with the corresponding trained one on the dataset


### Train with SeaDroneSee

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2201 tools/train.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_seadronesee.py --work-dir work_dir --num_gpu 4
```
> replace the work_dir with your customized one
### Test with SeaDroneSee

```

python -m torch.distributed.launch --nproc_per_node=1 --master_port=1001 ./tools/test.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_seadronesee.py yolov8_s_syncbn_fast_8xb16-500e_seadronessee_saliency_correctgridlarge_2400/epoch_50.pth```
> replace the checkpoint with the corresponding trained one on the dataset

```

## Acknowledgement

Zoomdet is an open source project, and is based on [mmyolo](https://github.com/open-mmlab/mmyolo)

## License

This project is released under the [Apache 2.0 license](LICENSE).
