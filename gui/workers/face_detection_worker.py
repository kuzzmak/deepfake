import torch

import numpy as np

import cv2 as cv

from PIL import Image

from enums import FACE_DETECTION_ALGORITHM, JOB_TYPE, MESSAGE_STATUS, MESSAGE_TYPE

from core.face_detection.algorithms.s3fd.data.config import cfg
from core.face_detection.algorithms.s3fd.s3fd import build_s3fd
from core.face_detection.algorithms.s3fd.utils.augmentations import to_chw_bgr

from gui.workers.worker import Worker

from message.message import Message, MessageBody, AnswerBody2

from utils import get_file_paths_from_dir

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class FaceDetectionWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, msg: Message):

        faces_directory, model_path, algorithm = msg.body.get_data()
        picture_paths = get_file_paths_from_dir(faces_directory)

        print(faces_directory)
        print(model_path)
        print(algorithm)

        if algorithm == FACE_DETECTION_ALGORITHM.S3FD:

            net = build_s3fd('test', cfg.NUM_CLASSES)
            net.load_state_dict(torch.load(
                model_path, map_location=torch.device('cpu')))
            net.eval()
            thresh = 0.6

            img_path = 'C:\\Users\\tonkec\\Documents\\deepfake\\core\\face_detection\\algorithms\\s3fd\\img\\test3.jpg'

            extraced_faces = detect(net, img_path, thresh)

            msg = Message(
                MESSAGE_TYPE.ANSWER,
                AnswerBody2(
                    status=MESSAGE_STATUS.OK,
                    job_type=JOB_TYPE.FACE_DETECTION,
                    finished=False,
                    data={
                        'faces': extraced_faces,
                    }
                )
            )


def detect(net, img_path, thresh):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv.INTER_LINEAR)
    # image = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = torch.autograd.Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv.imread(img_path, cv.IMREAD_COLOR)

    faces = []

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            # score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            left_up, right_down = (int(pt[0]), int(
                pt[1])), (int(pt[2]), int(pt[3]))
            j += 1
            faces.append((left_up, right_down))

    extracted_faces = extract_faces(faces, img)
    return extracted_faces

    # cv.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
    # conf = "{:.3f}".format(score)
    # point = (int(left_up[0]), int(left_up[1] - 5))
    # cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    # t2 = time.time()
    # print('detect:{} timer:{}'.format(img_path, t2 - t1))

    # cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


def extract_faces(faces, img):

    extracted_faces = []

    for face in faces:
        x1, y1, x2, y2 = face
        extracted_faces.append(img[y1: y2, x1: x2])

    return extract_faces
