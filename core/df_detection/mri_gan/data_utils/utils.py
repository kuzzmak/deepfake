import json
from glob import glob
import os
from pathlib import Path
import re
from typing import List, Match, Union

import cv2 as cv
import pandas as pd

from core.df_detection.mri_gan.utils import ConfigParser


def create_video_from_images(
    images,
    output_video_filename,
    fps=30,
    res=(
        1920,
        1080)):
    video = cv.VideoWriter(
        output_video_filename,
        cv.VideoWriter_fourcc(
            *"mp4v"),
        fps,
        res)
    for image in images:
        video.write(image)
    video.release()


def extract_images_from_video(input_video_filename, output_folder, res=None):
    os.makedirs(output_folder, exist_ok=True)
    capture = cv.VideoCapture(input_video_filename)
    frames_num = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        out_image_name = os.path.join(output_folder, "{}.jpg".format(i))
        if res is not None:
            frame = cv.resize(frame, res, interpolation=cv.INTER_AREA)
        cv.imwrite(out_image_name, frame, [cv.IMWRITE_JPEG_QUALITY, 100])


"""
sample entries from metadata.json of DFDC

{"iqqejyggsm.mp4": {"label": "FAKE", "split": "train", "original": "gzesfubacw.mp4"}
{"ooafcxxfrs.mp4": {"label": "REAL", "split": "train"}

"""


def get_dfdc_training_real_fake_pairs(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append(
                    (os.path.splitext(original)[0],
                     os.path.splitext(k)[0]))
    return pairs


def match_dfdc_dirs(directory: str) -> Union[Match, None]:
    return re.match(r'dfdc_(train|test|valid)_part_[0-9]+', directory)


def filter_dfdc_dirs(dirs) -> List[str]:
    matches = [match_dfdc_dirs(d) for d in dirs]
    matches = list(filter(lambda x: x is not None, matches))
    return [m.group(0) for m in matches]


def get_dfdc_training_video_filepaths(root_dir: Path) -> List[Path]:
    dirs = os.listdir(root_dir)
    dirs = filter_dfdc_dirs(dirs)
    dirs = [root_dir / d for d in dirs]
    json_paths = [d / 'metadata.json' for d in dirs]
    fps = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        for k, _ in metadata.items():
            fps.append(json_path.parent / k)
    return fps


# def get_dfdc_training_video_filepaths(root_dir) -> List[str]:
#     video_filepaths = []
#     for json_path in glob(os.path.join(root_dir, "metadata.json")):
#         pdir = Path(json_path).parent
#         with open(json_path, "r") as f:
#             metadata = json.load(f)
#         for k, v in metadata.items():
#             full_path = os.path.join(pdir, k)
#             video_filepaths.append(full_path)
#     return video_filepaths


def get_training_reals_and_fakes():
    root_dir = ConfigParser.getInstance().get_dfdc_train_data_path()
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append(k)
            else:
                originals.append(k)

    return originals, fakes


def get_valid_reals_and_fakes():
    labels_csv = ConfigParser.getInstance().get_dfdc_valid_label_csv_path()
    df = pd.read_csv(labels_csv, index_col=0)
    originals = list(df[df['label'] == 0].index.values)
    fakes = list(df[df['label'] == 1].index.values)

    return originals, fakes


def get_test_reals_and_fakes():
    labels_csv = ConfigParser.getInstance().get_dfdc_test_label_csv_path()
    df = pd.read_csv(labels_csv, index_col=0)
    originals = list(df[df['label'] == 0].index.values)
    fakes = list(df[df['label'] == 1].index.values)

    return originals, fakes


def get_video_frame_labels_mapping(cid, originals, fakes):
    cid_ = os.path.basename(cid)
    if cid_ in originals:
        crop_label = 0
    elif cid_ in fakes:
        crop_label = 1
    else:
        raise Exception('Unknown label')
    crop_items = glob(cid + '/*')
    df = pd.DataFrame(columns=['video_id', 'frame', 'label'])
    for crp_itm in crop_items:
        crp_itm_ = os.path.basename(crp_itm)
        new_row = {'video_id': cid_, 'frame': crp_itm_, 'label': crop_label}
        df = df.append(new_row, ignore_index=True)

    return df
