from core.aligner import Aligner
from core.dictionary import Dictionary
import cv2 as cv
from serializer.face_serializer import FaceSerializer
import numpy as np


if __name__ == '__main__':
    face_path = r'C:\Users\tonkec\Documents\deepfake\data\temp\trump_cut\metadata\frame_170.p'
    face = FaceSerializer.load(face_path)
    alignments_path = r'C:\Users\tonkec\Documents\deepfake\data\temp\trump_cut\metadata\alignments.json'
    alignments = Dictionary.load(alignments_path)
    alignment = alignments['frame_170.p']
    aligned = Aligner.align_image(
        face.raw_image.data,
        alignment,
        64,
    )
    alignment = np.vstack((alignment, [0, 0, 1]))
    alignment_inv = np.linalg.inv(alignment).astype(np.float32)[:2]
    s = face.raw_image.data.shape
    warped_revers = cv.warpAffine(aligned, alignment_inv, (100, 100))
    cv.imshow('reverse warp', warped_revers)
    # print(alignment_inv)
    # cv.imshow('image', face.detected_face)
    # cv.imshow('aligned', aligned)
    cv.waitKey()