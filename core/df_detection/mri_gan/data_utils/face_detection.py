from collections import OrderedDict
from glob import glob
import json
import multiprocessing
import os
from pathlib import Path
from typing import List
import torch
from torch.utils.data import DataLoader

import cv2 as cv
from facenet_pytorch.models.mtcnn import MTCNN
from PIL import Image
from tqdm import tqdm

from core.df_detection.mri_gan.data_utils.utils import (
    create_video_from_images,
    get_dfdc_training_video_filepaths,
)
from core.df_detection.mri_gan.utils import ConfigParser
from core.df_detection.mri_gan.data_utils.datasets import SimpleImageFolder


def get_face_detector_model(name='default'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # default is mtcnn model from facenet_pytorch
    if name == 'default':
        name = 'mtcnn'

    if name == 'mtcnn':
        detector = MTCNN(
            margin=0,
            thresholds=[0.85, 0.95, 0.95],
            device=device,
        )
    else:
        raise Exception("Unknown face detector model.")

    return detector


def locate_face_in_videofile(input_filepath=None, outfile_filepath=None):
    capture = cv.VideoCapture(input_filepath)
    frames_num = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv.CAP_PROP_FPS))

    detector = get_face_detector_model()
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        face_box = list(detector.detect(frame, landmarks=False))[0]
        if face_box is not None:
            for f in range(len(face_box)):
                fc = list(face_box[f])
                cv.rectangle(
                    frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
        frames.append(frame)
    create_video_from_images(
        frames,
        outfile_filepath,
        fps=org_fps,
        res=org_res)


def extract_landmarks_from_video(
    input_videofile: Path,
    out_dir: Path,
    batch_size=32,
    detector=None,
    overwrite=False,
    inference: bool = False
):
    # name of the video + .json for landmark file, e.g. aayrffkzxn.json
    name = input_videofile.stem + '.json'
    if inference:
        part = ''
    else:
        # part of the dataset, e.g. dfdc_train_part_2
        part = input_videofile.parts[-2]
    os.makedirs(out_dir / part, exist_ok=True)
    out_file = out_dir / part / name

    if not overwrite and out_file.is_file():
        return

    capture = cv.VideoCapture(str(input_videofile))
    frames_num = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    if detector is None:
        detector = get_face_detector_model()

    frames_dict = OrderedDict()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames_dict[i] = frame

    result = OrderedDict()
    batches = list()
    frames = list(frames_dict.values())
    num_frames_detected = len(frames)
    for i in range(0, num_frames_detected, batch_size):
        end = i + batch_size
        if end > num_frames_detected:
            end = num_frames_detected
        batches.append((list(range(i, end)), frames[i:end]))

    for j, frames_list in enumerate(batches):
        frame_indices, frame_items = frames_list
        batch_boxes, prob, keypoints = detector.detect(
            frame_items,
            landmarks=True,
        )
        batch_boxes = [
            b.tolist() if b is not None else None for b in batch_boxes
        ]
        keypoints = [k.tolist() if k is not None else None for k in keypoints]

        result.update({i: b for i, b in zip(
            frame_indices, zip(batch_boxes, keypoints))})

    with open(out_file, 'w') as f:
        json.dump(result, f)


def my_collate(batch):
    batch = zip(*batch)
    return batch


def extract_landmarks_from_images_batch(
        input_images_dir,
        landmarks_file,
        batch_size=1024,
        detector=None,
        overwrite=True):
    if not overwrite and os.path.isfile(landmarks_file):
        return

    if detector is None:
        detector = get_face_detector_model()

    imgdata = SimpleImageFolder(input_images_dir, transforms_=None)
    data_loader = DataLoader(
        imgdata,
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=my_collate,
    )
    result = dict()
    for img_list in tqdm(data_loader):
        imgs, img_names = img_list
        # imgs = imgs.cuda()
        batch_boxes, prob, keypoints = detector.detect(imgs, landmarks=True)

        b = len(img_names)

        for j in range(b):
            boxes = None if batch_boxes[j] is None else batch_boxes[j].tolist()
            lm = None if keypoints[j] is None else keypoints[j].tolist()
            data = [boxes, lm]
            result.update({os.path.basename(img_names[j]): data})

    with open(landmarks_file, "w") as f:
        json.dump(result, f)


def extract_landmarks_from_video_batch(
    file_paths: List[str],
    out_dir: str,
    num_processes: int = 2,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with multiprocessing.Pool(num_processes) as pool:
        jobs = []
        results = []
        for fp in file_paths:
            jobs.append(
                pool.apply_async(
                    extract_landmarks_from_video,
                    (fp, out_dir,),
                )
            )

        for job in tqdm(jobs, desc="Extracting landmarks"):
            results.append(job.get())


def draw_landmarks_on_video(in_videofile, out_videofile, landmarks_file):
    capture = cv.VideoCapture(in_videofile)
    frames_num = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    org_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    org_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    org_res = (org_width, org_height)
    org_fps = int(capture.get(cv.CAP_PROP_FPS))
    with open(landmarks_file, 'r') as jf:
        face_box_dict = json.load(jf)
    frames = list()
    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        if str(i) not in face_box_dict.keys():
            continue
        face_box = face_box_dict[str(i)]
        if face_box is not None:
            faces = face_box[0]
            lm = face_box[1]
            for f in range(len(faces)):
                fc = list(map(int, faces[f]))
                cv.rectangle(
                    frame, (fc[0], fc[1]), (fc[2], fc[3]), (0, 255, 0), 4)
                keypoints = lm[f]
                cv.circle(
                    frame, tuple(map(int, keypoints[0])),
                    2, (0, 155, 255),
                    2)
                cv.circle(
                    frame, tuple(map(int, keypoints[1])),
                    2, (0, 155, 255),
                    2)
                cv.circle(
                    frame, tuple(map(int, keypoints[2])),
                    2, (0, 155, 255),
                    2)
                cv.circle(
                    frame, tuple(map(int, keypoints[3])),
                    2, (0, 155, 255),
                    2)
                cv.circle(
                    frame, tuple(map(int, keypoints[4])),
                    2, (0, 155, 255),
                    2)

        frames.append(frame)
    create_video_from_images(frames, out_videofile, fps=org_fps, res=org_res)


def crop_faces_from_video(
    video_path: Path,
    landmarks_dir_path: Path,
    crop_faces_dir_path: Path,
    overwrite=False,
    frame_hops=10,
    buf=0.10,
    clean_up=True,
    inference: bool = False,
):
    name = video_path.stem
    name_metadata = name + '.json'
    if inference:
        part = ''
    else:
        part = video_path.parts[-2]
    landmarks_file = landmarks_dir_path / part / name_metadata
    out_dir = crop_faces_dir_path / part / name

    if not landmarks_file.is_file():
        return
    if not overwrite and out_dir.is_dir():
        return

    try:
        with open(landmarks_file, 'r') as jf:
            face_box_dict = json.load(jf)
    except Exception as e:
        print(f'Failed to parse landmark file: {str(landmarks_file)}.')
        print(e)
        raise e

    os.makedirs(out_dir, exist_ok=True)
    capture = cv.VideoCapture(str(video_path))
    frames_num = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        if i % frame_hops != 0:
            continue
        success, frame = capture.retrieve()
        if not success or str(i) not in face_box_dict:
            continue

        crops = []
        bboxes = face_box_dict[str(i)][0]
        if bboxes is None:
            continue
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = int(h * buf)
            p_w = int(w * buf)
            crop = frame[max(ymin - p_h, 0):ymax + p_h,
                         max(xmin - p_w, 0):xmax + p_w]
            crops.append(crop)

        for j, crop in enumerate(crops):
            cv.imwrite(os.path.join(out_dir, "{}_{}.png".format(i, j)), crop)

    capture.release()
    # if clean_up and os.path.isfile(in_videofile):
    #     os.remove(in_videofile)


def crop_faces_from_image(
        in_image_path,
        landmarks_dict,
        crop_faces_out_dir,
        overwrite=True,
        buf=0.10):
    in_image_name = os.path.basename(in_image_path)
    in_image_id = os.path.splitext(in_image_name)[0]
    os.makedirs(crop_faces_out_dir, exist_ok=True)
    out_dir = crop_faces_out_dir
    os.makedirs(out_dir, exist_ok=True)

    if not overwrite and os.path.isdir(out_dir):
        return

    try:
        lm = landmarks_dict
        image = cv.imread(in_image_path, cv.IMREAD_COLOR)
        crops = list()
        bboxes = lm[0]
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = int(h * buf)
            p_w = int(w * buf)
            crop = image[max(ymin - p_h, 0):ymax + p_h,
                         max(xmin - p_w, 0):xmax + p_w]
            crops.append(crop)

        for j, crop in enumerate(crops):
            cv.imwrite(
                os.path.join(
                    out_dir,
                    "{}_{}.png".format(
                        in_image_id,
                        j)),
                crop)

    except Exception as e:
        pass


def crop_faces_from_image_batch(
        input_images_dir,
        landmarks_file,
        crop_faces_out_dir):
    os.makedirs(crop_faces_out_dir, exist_ok=True)

    with open(landmarks_file, 'r') as jf:
        landmarks_dict = json.load(jf)

    input_filepath_list = glob(input_images_dir + '/*')
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        jobs = []
        results = []
        for input_filepath in tqdm(
                input_filepath_list,
                desc='Scheduling jobs'):
            in_image_name = os.path.basename(input_filepath)
            lm = landmarks_dict[in_image_name]
            jobs.append(
                pool.apply_async(
                    crop_faces_from_image,
                    (input_filepath, lm, crop_faces_out_dir,),))

        for job in tqdm(jobs, desc="Cropping faces from images"):
            results.append(job.get())


def crop_faces_from_video_batch(
        input_filepath_list,
        landmarks_path,
        crop_faces_out_dir):
    os.makedirs(crop_faces_out_dir, exist_ok=True)
    with multiprocessing.Pool(2) as pool:
        jobs = []
        results = []
        for input_filepath in input_filepath_list:
            jobs.append(
                pool.apply_async(
                    crop_faces_from_video,
                    (input_filepath,
                     landmarks_path,
                     crop_faces_out_dir,
                     ),
                ))

        for job in tqdm(jobs, desc="Cropping faces"):
            results.append(job.get())


def extract_landmarks_for_datasets():
    # DFDC dataset
    print(f'Extracting landmarks from DFDC train data')
    data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
    input_filepath_list = get_dfdc_training_video_filepaths(data_path_root)
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
    extract_landmarks_from_video_batch(input_filepath_list, landmarks_path)


def crop_faces_for_datasets():
    # DFDC dataset
    print(f'Cropping faces from DFDC train data')
    data_path_root = ConfigParser.getInstance().get_dfdc_train_data_path()
    input_filepath_list = get_dfdc_training_video_filepaths(data_path_root)
    landmarks_path = ConfigParser.getInstance().get_dfdc_landmarks_train_path()
    crops_path = ConfigParser.getInstance().get_dfdc_crops_train_path()
    crop_faces_from_video_batch(
        input_filepath_list,
        landmarks_path,
        crops_path,
    )
