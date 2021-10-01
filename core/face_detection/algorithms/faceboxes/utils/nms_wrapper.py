# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from core.face_detection.algorithms.faceboxes.utils.nms.cpu_nms import cpu_nms
# from core.face_detection.algorithms.faceboxes.utils.nms.gpu_nms import gpu_nms
from core.face_detection.algorithms.faceboxes.utils.nms.py_cpu_nms import py_cpu_nms

def nms(dets, thresh, device='cpu'):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    # if device == 'cpu':
    #     return cpu_nms(dets, thresh)
    # return gpu_nms(dets, thresh)
    return py_cpu_nms(dets, thresh)
