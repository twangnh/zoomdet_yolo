# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional

from mmdet.datasets import BaseDetDataset, CocoDataset

from ..registry import DATASETS, TASK_UTILS


class BatchShapePolicyDataset(BaseDetDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            ## for defect pcb
            # group_path = 'group'+data_info['img_path'].split('/')[-1][:5]+ '/' +data_info['img_path'].split('/')[-1][:5] + '/'
            # temp = data_info['img_path'].split('/')
            # temp.insert(-1, group_path)
            # data_info['img_path'] = '/'.join(temp)
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

@DATASETS.register_module()
class YOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    # METAINFO = {
    #     'classes':
    #     # ('pedestrian', 'people', 'bicycle',
    #     # 'car', 'van', 'truck', 'tricycle',
    #     # 'awning-tricycle', 'bus', 'motor'),
    #     # ('Dents', 'dent_marginal', 'dent_unacceptable',),
    #     ('ignored', 'swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy'),
    #     # palette is a list of color tuples, which is used for visualization.
    #     'palette':
    #     [(220, 20, 60)]
    # }
    pass

@DATASETS.register_module()
class YOLOv5VisdroneDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes':
        # ('pedestrian', 'people', 'bicycle',
        # 'car', 'van', 'truck', 'tricycle',
        # 'awning-tricycle', 'bus', 'motor'),
        # ('Dents', 'dent_marginal', 'dent_unacceptable',),
            ('pedestrian', 'people', 'bicycle',
             'car', 'van', 'truck', 'tricycle',
             'awning-tricycle', 'bus', 'motor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
    pass

@DATASETS.register_module()
class YOLOv5SeadroneseeDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes':
        # ('pedestrian', 'people', 'bicycle',
        # 'car', 'van', 'truck', 'tricycle',
        # 'awning-tricycle', 'bus', 'motor'),
        # ('Dents', 'dent_marginal', 'dent_unacceptable',),
        ('ignored', 'swimmer', 'boat', 'jetski', 'life_saving_appliances', 'buoy'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
    pass

@DATASETS.register_module()
class YOLOv5defectshampooDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes':
        # ('pedestrian', 'people', 'bicycle',
        # 'car', 'van', 'truck', 'tricycle',
        # 'awning-tricycle', 'bus', 'motor'),
        # ('Dents', 'dent_marginal', 'dent_unacceptable',),
        ('Dents', 'dent_marginal', 'dent_unacceptable',),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
    pass

@DATASETS.register_module()
class YOLOv5defectpcbDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes':
        # ('pedestrian', 'people', 'bicycle',
        # 'car', 'van', 'truck', 'tricycle',
        # 'awning-tricycle', 'bus', 'motor'),
        # ('Dents', 'dent_marginal', 'dent_unacceptable',),
        ('open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
    pass