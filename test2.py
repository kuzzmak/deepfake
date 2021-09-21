

from core.dataset.dataset import DeepfakeDataset
from torch.utils.data import dataset
from enums import DEVICE
from core.dataset.configuration import DatasetConfiguration
from core.model.original_ae import OriginalAE
from core.trainer.configuration import AdamConfiguration
import torch.nn as nn
from ignite.metrics import SSIM
import cv2 as cv
from torchvision import transforms
from tqdm import tqdm

from core.extractor import Extractor, ExtractorConfiguration


if __name__ == '__main__':

    # conf = ExtractorConfiguration(
    #     input_dir=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces',
    #     device=DEVICE.CUDA,
    # )
    # extractor = Extractor(conf)
    # extractor.run()

    # input_shape = (3, 128, 128)
    # device = DEVICE.CUDA

    # model = OriginalAE(input_shape).to(device.value)

    # optim_conf = AdamConfiguration(
    #     params=model.parameters(),
    #     learning_rate=1e-3,
    # )

    # data_transforms = transforms.Compose([transforms.ToTensor()])

    # dataset_conf = DatasetConfiguration(
    #     # faces_path=r'C:\Users\tonkec\Documents\deepfake\data\gen_faces\metadata',
    #     faces_path=r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces\metadata',
    #     input_shape=input_shape[1],
    #     batch_size=32,
    #     data_transforms=data_transforms,
    # )

    # pbar = tqdm(dataset_conf.data_loader)
    # for face in pbar:
    dat = DeepfakeDataset(
        r'C:\Users\kuzmi\Documents\deepfake\data\gen_faces\metadata', 128)
    for face in dat:
        aligned_face, aligned_mask = face
        cv.imshow('face', aligned_face)
        cv.imshow('mask', aligned_mask)
        cv.waitKey()
        break
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

    # tc = TrainerConfiguration(
    #     model=model,
    #     optim_conf=optim_conf,
    #     dataset_conf=dataset_conf,
    #     epochs=10,
    #     criterion=SSIM(data_range=1.0),
    # )

    # trainer = Trainer(tc)
