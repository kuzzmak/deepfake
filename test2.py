# import torch
# from torch.serialization import load
# from core.dataset.dataset import DeepfakeDataset
# from torch.utils.data import dataset
from dataclasses import dataclass
from gui.widgets.preview.configuration import PreviewConfiguration
from ssim import SSIM, ssim
import numpy as np
import cv2
from torch import optim
from torch.autograd import Variable
import torch
import threading
from enums import DEVICE
from core.dataset.configuration import DatasetConfiguration
from core.model.original_ae import OriginalAE
from core.trainer.configuration import AdamConfiguration
# import torch.nn as nn
# from ignite.metrics import SSIM
# import cv2 as cv
from torchvision import transforms
from tqdm import tqdm
from string import Template
from core.trainer.trainer import Trainer
from core.trainer.configuration import TrainerConfiguration
from torch.nn import MSELoss
# from ignite.metrics import SSIM
# from core.extractor import Extractor, ExtractorConfiguration
import PyQt5.QtWidgets as qwt
from test5 import DeepfakeTimeLapse
import sys
from common_structures import CommObject, TensorCommObject

import yaml
import logging.config

with open('logging_config.yaml', 'r') as f:
    _conf = yaml.safe_load(f.read())
logging.config.dictConfig(_conf)
logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)


if __name__ == '__main__':

    # npImg1 = cv2.imread("einstein.png")

    # img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
    # img2 = torch.rand(img1.size())

    # if torch.cuda.is_available():
    #     img1 = img1.cuda()
    #     img2 = img2.cuda()

    # img1 = Variable(img1,  requires_grad=False)
    # img2 = Variable(img2, requires_grad=True)

    # # Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
    # ssim_value = ssim(img1, img2)
    # print("Initial ssim:", ssim_value)

    # Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
    ssim_loss = SSIM()

    # optimizer = optim.Adam([img2], lr=0.01)

    # while ssim_value < 0.95:
    #     optimizer.zero_grad()
    #     ssim_out = -ssim_loss(img1, img2)
    #     ssim_value = - ssim_out
    #     print(ssim_value)
    #     ssim_out.backward()
    #     optimizer.step()

    device = DEVICE.CPU
    if torch.cuda.is_available():
        device = DEVICE.CUDA

    # # conf = ExtractorConfiguration(
    # #     input_dir=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces',
    # #     device=DEVICE.CUDA,
    # # )
    # # extractor = Extractor(conf)
    # # extractor.run()

    input_shape = (3, 128, 128)
    # # # device = DEVICE.CUDA

    model = OriginalAE(input_shape).to(device.value)

    optim_conf = AdamConfiguration(
        params=model.parameters(),
        learning_rate=5e-5,
        betas=(0.5, 0.999),
    )

    data_transforms = transforms.Compose([transforms.ToTensor()])

    dataset_conf = DatasetConfiguration(
        # faces_path=r'C:\Users\tonkec\Documents\deepfake\data\gen_faces\metadata',
        metadata_path=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces\metadata',
        input_shape=input_shape[1],
        batch_size=32,
        device=device,
        load_into_memory=True,
        data_transforms=data_transforms,
    )

    # desc_template = Template('epoch: $epoch, loss: $loss')

    # epoch = 1
    # desc = desc_template.substitute(
    #     epoch=epoch,
    #     loss=0,
    # )
    # pbar = tqdm(dataset_conf.data_loader, desc=desc)
    # for data in pbar:
    #     face, mask = data
    #     out = model(face)

    #     criterion = SSIM()
    #     loss = criterion(face[0].unsqueeze(0), out[0].unsqueeze(0))
    #     print(loss)
    #     break

    # desc = desc_template.substitute(
    #     epoch=epoch,
    #     loss=out.shape,
    # )
    # pbar.set_description(desc)

    #     break
    # dat = DeepfakeDataset(
    #     metadata_path=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces\metadata',
    #     input_shape=128,
    #     load_into_memory=True,
    #     device=DEVICE.CUDA,
    # )

    # for face in dat:
    #     aligned_face, aligned_mask = face
    #     print(aligned_face.device)
    #     break

    #     cv.imshow('face', aligned_face)
    #     cv.imshow('mask', aligned_mask)
    #     cv.waitKey()
    #     break
    # face = face.to(device.value)
    # out = model(face)
    # print(out)

    # for face in dataset_conf.data_loader:
    #     face = face.to('cuda')
    #     out = model(face)

    #     print(face.device)
    #     break
    #     print(out.shape)
    #     # print(face.shape)
    #     break

    tc = TrainerConfiguration(
        model=model,
        optim_conf=optim_conf,
        dataset_conf=dataset_conf,
        epochs=1,
        # criterion=SSIM(),
        criterion=MSELoss(),
        device=device,
        preview_conf=PreviewConfiguration(True, TensorCommObject()),
    )
    trainer = Trainer(tc)

    trainer_thread = threading.Thread(
        target=trainer.run,
        daemon=True,
    )
    trainer_thread.start()
    # trainer.run()

    app = qwt.QApplication(sys.argv)
    dftl = DeepfakeTimeLapse(tc.preview_conf.comm_object.data_sig)
    dftl.show()
    sys.exit(app.exec_())
