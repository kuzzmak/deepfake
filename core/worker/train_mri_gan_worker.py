import logging
import os
from pathlib import Path
import pickle
import pprint
import random
from typing import Optional

import numpy as np
import PyQt6.QtCore as qtc
from pytorch_msssim import SSIM
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from core.df_detection.mri_gan.mri_gan.dataset import MRIDataset
from core.df_detection.mri_gan.mri_gan.model import (
    Discriminator,
    GeneratorUNet,
    weights_init_normal,
)
from core.worker.worker import Worker
from configs.mri_gan_config import MRIGANConfig, print_line
from enums import DEVICE, JOB_NAME, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages

pp = pprint.PrettyPrinter(indent=4)


def denormalize(img):
    img = (img + 1) / 2  # [-1, 1] => [0, 1]
    return img * 255


logger = logging.getLogger(__name__)


class TrainMRIGANWorker(Worker):

    def __init__(
        self,
        image_size: int,
        batch_size: int,
        lr: float,
        epochs: int,
        device: DEVICE = DEVICE.CPU,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._image_size = image_size
        self._batch_size = batch_size
        self._lr = lr
        self._epochs = epochs
        self._device = device
        self._log_dir = Path(MRIGANConfig.get_instance().get_log_dir_name())

    def run_job(self) -> None:
        m_p = MRIGANConfig.get_instance().get_mri_gan_model_params()
        device = self._device.value

        data_transforms = torchvision.transforms.Compose([
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Loss functions
        criterion_GAN = torch.nn.MSELoss().to(device)
        criterion_pixelwise = torch.nn.MSELoss().to(device)
        criterion_ssim = SSIM(
            win_size=11,
            win_sigma=1.5,
            data_range=255,
            size_average=True,
            channel=3,
            nonnegative_ssim=True,
        ).to(device)

        # Calculate output of image discriminator (PatchGAN)
        patch = (
            1,
            self._image_size // 2 ** 4,
            self._image_size // 2 ** 4,
        )

        # Initialize generator and discriminator
        generator = GeneratorUNet().to(device)
        discriminator = Discriminator().to(device)

        # Optimizers
        optimizer_G = torch.optim.Adam(
            generator.parameters(),
            lr=self._lr,
            betas=(m_p['b1'], m_p['b2']),
        )
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=self._lr,
            betas=(m_p['b1'], m_p['b2']),
        )
        losses = []
        # ssim_report = []
        global_batches_done = 0
        start_epoch = 0
        mri_gan_metadata = dict()
        loss_D_lowest = float('inf')
        loss_G_lowest = float('inf')

        # if train_resume_dir is not None:
        #     checkpoint_path = train_resume_dir
        #     print(f'Loading state dict {checkpoint_path}')
        #     checkpoint = torch.load(checkpoint_path)
        #     generator.load_state_dict(checkpoint['generator_state_dict'])
        #     discriminator.load_state_dict(
        #         checkpoint['discriminator_state_dict'])
        #     optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        #     optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        #     log_dir = checkpoint['log_dir']
        #     start_epoch = checkpoint['epoch'] + 1
        #     print(f'Override log dir {log_dir}')
        #     losses_file = os.path.join(
        #         log_dir, model_params['model_name'],
        #         model_params['losses_file'])
        #     metadata_file = os.path.join(
        #         log_dir, model_params['model_name'],
        #         model_params['metadata_file'])
        #     ssim_report_file = os.path.join(
        #         log_dir, model_params['model_name'],
        #         model_params['ssim_report_file'])

        #     losses = pickle.load(open(losses_file, "rb"))
        #     ssim_report = pickle.load(open(ssim_report_file, "rb"))
        #     mri_gan_metadata = pickle.load(open(metadata_file, "rb"))
        #     global_batches_done = mri_gan_metadata['global_batches_done']
        #     if len(losses) != global_batches_done + 1:
        #         print(
        #             f'losses len = {len(losses)}, ' +
        #             f'mri_gan_metadata = {mri_gan_metadata}'
        #         )
        #         print(f'Use data from metadata file')
        #         losses = losses[0:global_batches_done]
        #         # raise Exception('Bad metadata and saved states')
        #     loss_D_lowest = mri_gan_metadata['loss_D_lowest']
        #     loss_G_lowest = mri_gan_metadata['loss_G_lowest']
        # else:
        logger.debug('Initializing weights.')
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        checkpoint_path = self._log_dir / \
            m_p['model_name'] / 'checkpoint.chkpt'
        losses_file = self._log_dir / \
            m_p['model_name'] / m_p['losses_file']
        metadata_file = self._log_dir / \
            m_p['model_name'] / m_p['metadata_file']
        # ssim_report_file = self._log_dir / \
        #     m_p['model_name'] / m_p['ssim_report_file']

        m_p['log_dir'] = self._log_dir
        checkpoint_best_G_path = self._log_dir / \
            m_p['model_name'] / 'checkpoint_best_G.chkpt'
        checkpoint_best_D_path = self._log_dir / \
            m_p['model_name'] / 'checkpoint_best_D.chkpt'
        generated_samples_path = self._log_dir / \
            m_p['model_name'] / 'generated_samples'
        os.makedirs(generated_samples_path, exist_ok=True)

        print_line()
        print('model_params')
        pp.pprint(m_p)
        print_line()

        self.running.emit()

        conf_wgt_msg = Messages.CONFIGURE_WIDGET(
            SIGNAL_OWNER.TRAIN_MRI_GAN_WORKER,
            WIDGET.JOB_PROGRESS,
            'setMaximum',
            [self._epochs],
            JOB_NAME.TRAIN_MRI_GAN,
        )
        self.send_message(conf_wgt_msg)

        for e in range(self._epochs):

            if self.should_exit():
                logger.info('Received stop signal, exiting now.')
                return

            if e < start_epoch:
                logger.debug(f'Skipping epoch {e}.')
                continue

            # we are creating dataloader at each epoch as we want to sample new
            # fake image randomly as each epoch. We have lesser real image and
            # more fake. We can get more real images but that would impose a
            # lot of computation power need considering the dataset size we
            # have.

            logger.debug(f'Creating data loaders for epoch: {e}.')
            train_dataset = MRIDataset(
                mode='train',
                transforms=data_transforms,
                frac=m_p['frac'],
            )
            test_dataset = MRIDataset(
                mode='test',
                transforms=data_transforms,
                frac=m_p['frac'],
            )

            num_workers = 8
            # num_workers = 0
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self._batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            desc = '[e:{}/{}] [G_loss:{}] [D_loss:{}]' \
                .format(e, self._epochs, 'N/A', 'N/A')
            pbar = tqdm(train_dataloader, desc=desc)

            for local_batch_num, batch in enumerate(pbar):

                if self.should_exit():
                    logger.info('Received stop signal, exiting now.')
                    return

                generator.train()
                discriminator.train()
                real_A = batch['A'].to(device)
                real_B = batch['B'].to(device)

                valid = torch.ones((real_A.size(0), *patch)).to(device)
                fake = torch.zeros((real_A.size(0), *patch)).to(device)

                #  Train Generator
                optimizer_G.zero_grad()

                # GAN loss
                fake_B = generator(real_A)
                pred_fake = discriminator(fake_B, real_A)
                loss_GAN = criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_B, real_B)
                fake_B_dn = denormalize(fake_B)
                real_B_dn = denormalize(real_B)
                # SSIM loss
                loss_ssim = torch.sqrt(
                    1 - criterion_ssim(fake_B_dn, real_B_dn)
                )
                # Total generator loss
                loss_G = loss_GAN + m_p['lambda_pixel'] * (
                    m_p['tau'] * loss_pixel +
                    (1 - m_p['tau']) * loss_ssim)

                loss_G.backward()
                optimizer_G.step()

                #  Train Discriminator
                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(real_B, real_A)
                loss_real = criterion_GAN(pred_real, valid)
                # Fake loss
                pred_fake = discriminator(fake_B.detach(), real_A)
                loss_fake = criterion_GAN(pred_fake, fake)
                # Total discriminator loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()

                desc = '[e:{}/{}] [G_loss:{}] [D_loss:{}]' \
                    .format(e, self._epochs, loss_G.item(), loss_D.item())
                pbar.set_description(desc=desc, refresh=True)

                losses.append(
                    [
                        e,
                        local_batch_num,
                        global_batches_done,
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_pixel.item(),
                        loss_ssim.item(),
                        loss_D.item(),
                        loss_real.item(),
                        loss_fake.item(),
                    ]
                )

                mri_gan_metadata['global_batches_done'] = global_batches_done
                mri_gan_metadata['model_params'] = m_p

                global_batches_done += 1

                if global_batches_done % m_p['sample_gen_freq'] == 0:
                    try:
                        generator.eval()
                        imgs = next(iter(test_dataloader))
                        rand_start = random.randint(
                            0,
                            self._batch_size -
                            m_p['test_sample_size'],
                        )
                        rand_end = rand_start + \
                            m_p['test_sample_size']
                        real_A = imgs['A'][rand_start:rand_end].to(device)
                        real_B = imgs['B'][rand_start:rand_end].to(device)
                        fake_B = generator(real_A)
                        img_sample = torch.cat(
                            (real_A.data, real_B.data, fake_B.data),
                            -2,
                        )
                        os.makedirs(
                            generated_samples_path / str(e),
                            exist_ok=True,
                        )
                        img_path = '{}/{}/{}.png'.format(
                            str(generated_samples_path),
                            e,
                            local_batch_num,
                        )
                        save_image(
                            img_sample,
                            img_path,
                            nrow=int(np.sqrt(m_p['test_sample_size'])),
                            normalize=True,
                        )

                        # mean_ssim = get_ssim_report(
                        #     global_batches_done - 1, model_params, imgs,
                        #     generator, device, save_img=False)
                        # ssim_report.append(
                        #     [e, local_batch_num, global_batches_done,
                        # mean_ssim])
                        # pickle.dump(ssim_report, open(ssim_report_file,
                        # "wb"))

                        # generate_graphs(
                        #     losses_file, ssim_report_file, model_params)
                    except Exception as expn:
                        print(f'Exception {expn}')
                        pass

                if global_batches_done % m_p['chkpt_freq'] == 0:
                    check_point_dict = {
                        'epoch': e,
                        'model_params': m_p,
                        'log_dir': str(self._log_dir),
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                    }
                    torch.save(check_point_dict, checkpoint_path)

                if loss_D.item() < loss_D_lowest:
                    loss_D_lowest = loss_D.item()

                    check_point_dict = {
                        'epoch': e,
                        'model_params': m_p,
                        'log_dir': str(self._log_dir),
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                    }
                    torch.save(check_point_dict, checkpoint_best_D_path)

                if loss_G.item() < loss_G_lowest:
                    loss_G_lowest = loss_G.item()

                    check_point_dict = {
                        'epoch': e,
                        'model_params': m_p,
                        'log_dir': str(self._log_dir),
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                    }
                    torch.save(check_point_dict, checkpoint_best_G_path)

                mri_gan_metadata['loss_D_lowest'] = loss_D_lowest
                mri_gan_metadata['loss_G_lowest'] = loss_G_lowest
                pickle.dump(losses, open(losses_file, 'wb'))
                pickle.dump(mri_gan_metadata, open(metadata_file, 'wb'))

            self.report_progress(
                SIGNAL_OWNER.LANDMARK_EXTRACTION_WORKER,
                JOB_TYPE.TRAIN_MRI_GAN,
                e,
                self._epochs,
            )

        logger.info('MRI GAN training finished.')
