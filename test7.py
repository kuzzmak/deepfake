import torch
from config import _FaceDetectionAlgorithms
from core.extractor import ExtractorConfiguration
from core.face_alignment.face_aligner import FaceAligner
from core.face_alignment.utils import get_face_mask
from core.face_detection.algorithms.faceboxes.faceboxes_fdm import FaceboxesFDM
from core.image.image import Image
from core.landmark_detection.algorithms.fan.fan_ldm import FANLDM
from core.model.original_ae import OriginalAE
from enums import DEVICE
import cv2 as cv
from torchvision import transforms
import numpy as np
from PIL import Image as IMG
from serializer.face_serializer import FaceSerializer

from utils import tensor_to_np_image


if __name__ == '__main__':
    model_path = r'C:\Users\\kuzmi\\Documents\\deepfake\\models\best_model_0.pt'
    device = DEVICE.CPU
    if torch.cuda.is_available():
        device = DEVICE.CUDA
    # print('Loading model')
    # model = OriginalAE((3, 64, 64))
    # weights = torch.load(model_path, map_location=device.value)
    # model.load_state_dict(weights)
    # model.to(device.value)
    # model.eval()
    # print('Model loaded')

    # print('Preparing face')
    # face_path = r'C:\Users\kuzmi\Desktop\frame_224_1.jpg'
    # img = Image.load(face_path)
    # fdm = FaceboxesFDM(device)
    # face = fdm.detect_faces(img)[0]
    # face.raw_image = img
    # ldm = FANLDM(device)
    # landmarks = ldm.detect_landmarks(face)
    # face.landmarks = landmarks
    # face.mask = get_face_mask(face.raw_image.data, landmarks.dots)
    # FaceAligner.align_face(face, 64)
    # print('Face preparation done')

    # FaceSerializer.save(face, 'inface.p')

    # print('Inference')
    # img_ten = transforms.ToTensor()(face.aligned_image)
    # img_ten = img_ten.unsqueeze(0)
    # img_ten = img_ten.to(device.value)
    # y_pred_A_A, _, y_pred_A_B, _ = model(img_ten)
    # print('Inference done')

    # pred = y_pred_A_B.squeeze(0)
    # pred = tensor_to_np_image(pred).astype(np.uint8)
    # IMG.fromarray(pred).save('prediction.jpg')
    # cv.imshow('im', pred)
    # cv.waitKey()
    # cv.imshow('im', face.aligned_image)
    # cv.waitKey()

    f_p = r'C:\Users\kuzmi\Documents\deepfake\inface.p\metadata_frame_224_1.p'
    face = FaceSerializer.load(f_p)
    p_p = r'C:\Users\kuzmi\Documents\deepfake\prediction.jpg'
    pred = cv.imread(p_p, cv.IMREAD_COLOR)

    cv.imshow('raw', face.raw_image.data)
    cv.imshow('prediction', pred)
    cv.imshow('aligned', face.aligned_image)
    cv.waitKey()
