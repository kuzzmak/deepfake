import multiprocessing
import os
import random
from typing import Optional

import numpy as np
import PIL
import PyQt6.QtCore as qtc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.mri_gan_config import MRIGANConfig, print_green
from core.df_detection.mri_gan.deep_fake_detect.DeepFakeDetectModel import \
    DeepFakeDetectModel
from core.df_detection.mri_gan.deep_fake_detect.datasets import \
    DFDCDatasetSimple
from core.df_detection.mri_gan.deep_fake_detect.training import (
    valid_epoch,
)
from core.df_detection.mri_gan.deep_fake_detect.utils import (
    ENCODER_PARAMS,
    save_all_model_results,
)
from core.worker import Worker
from enums import DEVICE, MODE, MRI_GAN_DATASET


def get_predictions(output):
    return torch.round(torch.sigmoid(output))


def get_probability(output):
    return torch.sigmoid(output)


class TrainDeepfakeDetectorWorker(Worker):

    def __init__(
        self,
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._device = device

    def run_job(self) -> None:
        device = self._device.value
        model_params = MRIGANConfig \
            .get_instance() \
            .get_deep_fake_training_params()

        #
        # model_params['batch_format'] = 'simple' -> generates batches as below
        # batch_size x color_channel x frame_height x frame-width
        #

        encoder_name = MRIGANConfig \
            .get_instance() \
            .get_default_cnn_encoder_name()
        imsize = ENCODER_PARAMS[encoder_name]["imsize"]
        model_params['encoder_name'] = encoder_name
        model_params['imsize'] = imsize

        def gaussian_blur(img):
            return img.filter(PIL.ImageFilter.BoxBlur(
                random.choice([1, 2, 3])
            ))

        if model_params['train_transform'] == 'complex':
            train_transform = transforms.Compose([
                transforms.RandomChoice([
                    transforms.RandomCrop(imsize, imsize),
                    transforms.ColorJitter(contrast=random.random()),
                    transforms.Lambda(gaussian_blur),
                ]),
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.ColorJitter(brightness=random.random()),
                    transforms.RandomRotation(30),
                ]),
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(0.30),
            ])

        elif model_params['train_transform'] == 'simple':
            train_transform = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(0.30),
            ])
        else:
            raise Exception("model_params['train_transform'] not supported")

        valid_transform = transforms.Compose([
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        num_workers = multiprocessing.cpu_count() - 2
        # num_workers = 0
        train_dataset = DFDCDatasetSimple(
            mode=MODE.TRAIN,
            transform=train_transform,
            data_size=MRIGANConfig.get_instance().get_training_sample_size(),
            dataset=MRI_GAN_DATASET[model_params['dataset'].upper()],
            label_smoothing=model_params['label_smoothing'],
        )
        valid_dataset = DFDCDatasetSimple(
            mode=MODE.VALID,
            transform=valid_transform,
            data_size=MRIGANConfig.get_instance().get_valid_sample_size(),
            dataset=MRI_GAN_DATASET[model_params['dataset'].upper()],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=model_params['batch_size'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=model_params['batch_size'],
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        print(f"Batch_size {model_params['batch_size']}")
        model = DeepFakeDetectModel(
            frame_dim=model_params['imsize'],
            encoder_name=model_params['encoder_name'],
        ).to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_params['learning_rate'],
        )
        return
        # if model_params['fp16']:
        #     model, optimizer = amp.initialize(
        #         model, optimizer,
        #         opt_level=model_params['opt_level'],
        #         loss_scale='dynamic',
        #     )
        print(f'model params {model_params}')
        start_epoch = 0
        lowest_v_epoch_loss = float('inf')
        highest_v_epoch_acc = 0.0
        model_train_accuracies = []
        model_train_losses = []
        model_valid_accuracies = []
        model_valid_losses = []

        train_resume_checkpoint = None
        if train_resume_checkpoint is not None:
            # saved_epoch, model, optimizer, _, log_dir, amp_dict = load_checkpoint(
            #     model,
            #     optimizer,
            #     train_resume_checkpoint,
            # )
            # if model_params['fp16']:
            #     amp.load_state_dict(amp_dict)
            # start_epoch = saved_epoch + 1
            # print(f'Resuming Training from epoch {start_epoch}')
            # if os.path.basename(log_dir) == 'highest_acc' or os.path.basename(
            #         log_dir) == 'lowest_loss':
            #     log_dir = Path(log_dir).parent

            # print(f'Resetting log_dir to {log_dir}')

            # model_type = train_resume_checkpoint.split('/')[-2]
            # model_train_accuracies, model_train_losses, model_valid_accuracies, \
            #     model_valid_losses = load_acc_loss(model, log_dir, model_type)

            # if len(model_train_accuracies) != start_epoch:
            #     raise Exception(
            #         f'Error! model_train_accuracies = {model_train_accuracies} len is {len(model_train_accuracies)}, expected {start_epoch}')
            # if len(model_train_losses) != start_epoch:
            #     raise Exception(
            #         f'Error! model_train_losses = {model_train_losses}')
            # if len(model_valid_accuracies) != start_epoch:
            #     raise Exception(
            #         f'Error! model_valid_accuracies = {model_valid_accuracies}')
            # if len(model_valid_losses) != start_epoch:
            #     raise Exception(
            #         f'Error! model_valid_losses = {model_valid_losses}')
            # lowest_v_epoch_loss = min(model_valid_losses)
            # highest_v_epoch_acc = max(model_valid_accuracies)
            print('Loaded model acc and losses data successfully')

        else:
            print('Starting training from scratch')

        desc = 'Training[Train Acc={:02.4f}%|Loss Total={:.8f} Fake={:.8f} '
        desc += 'Real={:.8f}][Val Acc={:02.4f}%|Loss Total={:.8f} Fake={:.8f} '
        desc += 'Real={:.8f}]'
        tqdm_train_descr = desc.format(
            0,
            float('inf'),
            float('inf'),
            float('inf'),
            0,
            float('inf'),
            float('inf'),
            float('inf'),
        )
        tqdm_train_obj = tqdm(
            range(model_params['epochs']),
            desc=tqdm_train_descr,
        )

        log_dir = MRIGANConfig.get_instance().get_log_dir_name()
        train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'runs'))

        self.running.emit()

        for e in tqdm_train_obj:

            if e < start_epoch:
                print(f"Skipping epoch {e}")
                continue

            train_valid_use_tqdm = True
            (
                model,
                t_epoch_accuracy,
                t_epoch_loss,
                t_epoch_fake_loss,
                t_epoch_real_loss,
            ) = train_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                model_params['batch_size'],
                device,
                model_params,
            )
            model_train_accuracies.append(t_epoch_accuracy)
            model_train_losses.append(t_epoch_loss)

            tqdm_descr = desc.format(
                t_epoch_accuracy,
                t_epoch_loss,
                t_epoch_fake_loss,
                t_epoch_real_loss,
                0,
                float('inf'),
                float('inf'),
                float('inf'),
            )
            tqdm_train_obj.set_description(tqdm_descr)
            tqdm_train_obj.update()

            train_writer.add_scalar(
                'Training: loss per epoch', t_epoch_loss, e)
            train_writer.add_scalar(
                'Training: fake loss per epoch',
                t_epoch_fake_loss,
                e,
            )
            train_writer.add_scalar(
                'Training: real loss per epoch',
                t_epoch_real_loss,
                e,
            )
            train_writer.add_scalar(
                'Training: accuracy per epoch',
                t_epoch_accuracy,
                e,
            )

            print_green(tqdm_descr)
            (
                v_epoch_accuracy,
                v_epoch_loss,
                v_epoch_fake_loss,
                v_epoch_real_loss,
                all_predicted_labels,
                all_ground_truth_labels,
                all_filenames,
                probabilities,
            ) = valid_epoch(
                epoch=e,
                model=model,
                criterion=criterion,
                data_loader=valid_loader,
                batch_size=model_params['batch_size'],
                device=device,
                log_dir=log_dir,
                sum_writer=train_writer,
                use_tqdm=train_valid_use_tqdm,
                model_params=model_params,
            )

            model_valid_accuracies.append(v_epoch_accuracy)
            model_valid_losses.append(v_epoch_loss)

            tqdm_descr = desc.format(
                t_epoch_accuracy, t_epoch_loss, t_epoch_fake_loss,
                t_epoch_real_loss, v_epoch_accuracy, v_epoch_loss,
                v_epoch_fake_loss, v_epoch_real_loss)
            tqdm_train_obj.set_description(tqdm_descr)
            tqdm_train_obj.update()

            train_writer.add_scalar(
                'Validation: loss per epoch', v_epoch_loss, e)
            train_writer.add_scalar(
                'Validation: fake loss per epoch',
                v_epoch_fake_loss,
                e)
            train_writer.add_scalar(
                'Validation: real loss per epoch',
                v_epoch_real_loss,
                e)
            train_writer.add_scalar(
                'Validation: accuracy per epoch',
                v_epoch_accuracy,
                e)

            print_green(tqdm_descr)
            print(
                f'Saving model results at {log_dir}/latest_epoch for epoch {e}')
            # if model_params['fp16']:
            #     amp_dict = amp.state_dict()
            # else:
            amp_dict = None
            save_all_model_results(
                model=model,
                model_params=model_params,
                optimizer=optimizer,
                criterion=criterion.__class__.__name__,
                train_losses=model_train_losses,
                train_accuracies=model_train_accuracies,
                valid_losses=model_valid_losses,
                valid_accuracies=model_valid_accuracies,
                valid_predicted=all_predicted_labels,
                valid_ground_truth=all_ground_truth_labels,
                valid_sample_names=all_filenames,
                epoch=e,
                log_dir=log_dir,
                log_kind='latest_epoch',
                probabilities=probabilities,
                amp_dict=amp_dict,
            )

            if v_epoch_loss < lowest_v_epoch_loss:
                lowest_v_epoch_loss = v_epoch_loss
                print_green(
                    f'Saving best model (low loss) results at {log_dir}/lowest_loss for epoch {e}')
                # if model_params['fp16']:
                #     amp_dict = amp.state_dict()
                # else:
                amp_dict = None
                save_all_model_results(
                    model=model,
                    model_params=model_params,
                    optimizer=optimizer,
                    criterion=criterion.__class__.__name__,
                    train_losses=model_train_losses,
                    train_accuracies=model_train_accuracies,
                    valid_losses=model_valid_losses,
                    valid_accuracies=model_valid_accuracies,
                    valid_predicted=all_predicted_labels,
                    valid_ground_truth=all_ground_truth_labels,
                    valid_sample_names=all_filenames,
                    epoch=e,
                    log_dir=log_dir,
                    log_kind='lowest_loss',
                    probabilities=probabilities,
                    amp_dict=amp_dict)

            if highest_v_epoch_acc < v_epoch_accuracy:
                highest_v_epoch_acc = v_epoch_accuracy
                print_green(
                    f'Saving best model (high acc) results at {log_dir}/highest_acc for epoch {e}')
                # if model_params['fp16']:
                #     amp_dict = amp.state_dict()
                # else:
                amp_dict = None
                save_all_model_results(
                    model=model,
                    model_params=model_params,
                    optimizer=optimizer,
                    criterion=criterion.__class__.__name__,
                    train_losses=model_train_losses,
                    train_accuracies=model_train_accuracies,
                    valid_losses=model_valid_losses,
                    valid_accuracies=model_valid_accuracies,
                    valid_predicted=all_predicted_labels,
                    valid_ground_truth=all_ground_truth_labels,
                    valid_sample_names=all_filenames,
                    epoch=e,
                    log_dir=log_dir,
                    log_kind='highest_acc',
                    probabilities=probabilities,
                    amp_dict=amp_dict,
                )

        return model, model_params, criterion, log_dir


def train_epoch(
    model: DeepFakeDetectModel,
    criterion,
    optimizer,
    data_loader: DataLoader,
    batch_size: int,
    device,
    model_params,
):
    losses = []
    fake_losses = []
    real_losses = []
    accuracies = []
    total_samples = 0
    total_correct = 0
    model.train(True)

    train_data_iter = data_loader

    for batch_id, samples in enumerate(train_data_iter):
        # prepare data before passing to model
        optimizer.zero_grad()

        frames = samples['frame_tensor'].to(device)
        labels = samples['label'].to(device).unsqueeze(1)
        batch_size = labels.shape[0]

        output = model(frames)
        labels = labels.type_as(output)
        fake_loss = 0
        real_loss = 0
        fake_idx = labels > 0.5
        real_idx = labels <= 0.5
        if torch.sum(fake_idx * 1) > 0:
            fake_loss = criterion(output[fake_idx], labels[fake_idx])
        if torch.sum(real_idx * 1) > 0:
            real_loss = criterion(output[real_idx], labels[real_idx])

        batch_loss = (fake_loss + real_loss) / 2

        batch_loss_val = batch_loss.item()
        real_loss_val = 0 if real_loss == 0 else real_loss.item()
        fake_loss_val = 0 if fake_loss == 0 else fake_loss.item()

        # if model_params['fp16']:
        #     with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        batch_loss.backward()

        optimizer.step()

        predicted = get_predictions(output).to('cpu').detach().numpy()
        labels = labels.to('cpu').detach().numpy()
        if model_params['label_smoothing'] != 0:
            labels = labels.round()
        batch_corr = (predicted == labels).sum().item()

        total_samples += batch_size
        total_correct += batch_corr
        losses.append(batch_loss_val)
        fake_losses.append(fake_loss_val)
        real_losses.append(real_loss_val)

        batch_accuracy = batch_corr * 100 / batch_size
        accuracies.append(batch_accuracy)

    return (
        model,
        np.mean(accuracies),
        np.mean(losses),
        np.mean(fake_losses),
        np.mean(real_losses),
    )
