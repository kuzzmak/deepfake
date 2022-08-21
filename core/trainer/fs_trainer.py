import random
from typing import List

import torch
import torch.nn.functional as F
import torchvision
import wandb

from core.trainer.trainer import StepTrainer, StepTrainerConfiguration
from variables import IMAGENET_MEAN, IMAGENET_STD


class FSTrainer(StepTrainer):
    """Trainer implementation for the face swapping model which uses GANs.
    """

    def __init__(
        self,
        conf: StepTrainerConfiguration,
    ) -> None:
        super().__init__(conf)

        self._conf = conf

        meters = [
            'loss_Gmain',
            'loss_G_ID',
            'loss_G_Rec',
            'feat_match_loss',
            'loss_Dgen',
            'loss_Dreal',
            'loss_D',
        ]
        self.register_meters(meters)

        self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        self._imagenet_std = torch.Tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._imagenet_mean = torch.Tensor(IMAGENET_STD).view(3, 1, 1)

        self._lambda_id = self._conf.model_config.options['lambda_id']
        self._lambda_feat = self._conf.model_config.options['lambda_feat']

    def post_model_init(self) -> None:
        self._optim_g = self._model.optimizer_G
        self._optim_d = self._model.optimizer_D
        self._model.netD.feature_network.requires_grad_(False)

    def post_init_logging(self) -> None:
        if self._conf.wandb:
            wandb.config.update({'steps': self._conf.steps})

    def save_checkpoint(self) -> None:
        save_path = self._checkpoint_dir / \
            f'step_{self._current_step + 1}_checkpoint.pt'
        torch.save(
            {
                'netG': self._model.netG.state_dict(),
                'netD': self._model.netD.state_dict(),
                'optim_g': self._optim_g.state_dict(),
                'optim_d': self._optim_d.state_dict(),
                'current_step': self._current_step + 1,
            },
            save_path,
        )
        with open(self.get_latest_checkpoints_file_path(), 'wt') as f:
            f.write(f'latest_checkpoint: {str(save_path)}')

    def load_checkpoint(self) -> None:
        chkpt_fp = self.get_latest_checkpoints_file_path()
        if not chkpt_fp.exists():
            print(
                'file with latest checkpoint does not exist, ' +
                'training from scratch'
            )
            return
        with open(chkpt_fp, 'r') as f:
            latest = f.read().split(':')[1].strip()
        checkpoint = torch.load(
            latest,
            map_location=lambda storage, loc: storage,
        )
        self._model.netG.load_state_dict(checkpoint['netG'])
        self._model.netD.load_state_dict(checkpoint['netD'])
        self._optim_g.load_state_dict(checkpoint['optim_g'])
        self._optim_d.load_state_dict(checkpoint['optim_d'])
        current_step = checkpoint['current_step']
        self._starting_step = current_step

    @staticmethod
    def _prepare_for_arcface_112(img: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            img,
            size=(112, 112),
            mode='bicubic',
        )

    def _make_latent_id(self, images: torch.Tensor) -> torch.Tensor:
        img_id_112 = FSTrainer._prepare_for_arcface_112(images)
        latent_id = self._model.netArc(img_id_112)
        return F.normalize(latent_id, p=2, dim=1)

    def train_one_step(self) -> None:
        self._model.netG.train()

        for idx in range(2):
            img_1, img_2 = self.get_batch_of_data()
            img_1: torch.Tensor = img_1.to(self._device)
            img_2: torch.Tensor = img_2.to(self._device)
            randindex = list(range(self._conf.batch_size))
            random.shuffle(randindex)
            if self._current_step % 2 == 0:
                img_id = img_2
            else:
                img_id = img_2[randindex]

            latent_id = self._make_latent_id(img_id)

            if idx:
                img_fake = self._model.netG(img_1, latent_id)
                gen_logits, _ = self._model.netD(img_fake.detach(), None)
                loss_Dgen = (
                    F.relu(torch.ones_like(gen_logits) + gen_logits)
                ).mean()
                self.update_meter('loss_Dgen', loss_Dgen.item())

                real_logits, _ = self._model.netD(img_2, None)
                loss_Dreal = (
                    F.relu(torch.ones_like(real_logits) - real_logits)
                ).mean()
                self.update_meter('loss_Dreal', loss_Dreal.item())

                loss_D = loss_Dgen + loss_Dreal
                self.update_meter('loss_D', loss_D.item())

                self._optim_d.zero_grad()
                loss_D.backward()
                self._optim_d.step()
            else:
                img_fake = self._model.netG(img_1, latent_id)
                gen_logits, feat = self._model.netD(img_fake, None)
                loss_Gmain = (-gen_logits).mean()
                self.update_meter('loss_Gmain', loss_Gmain.item())

                latent_fake = self._make_latent_id(img_fake)
                loss_G_ID = (1 - self._cos(latent_fake, latent_id)).mean()
                self.update_meter('loss_G_ID', loss_G_ID.item())

                real_feat = self._model.netD.get_feature(img_1)
                feat_match_loss = self._model.criterionFeat(
                    feat["3"],
                    real_feat["3"],
                )
                self.update_meter('feat_match_loss', feat_match_loss.item())

                loss_G = loss_Gmain \
                    + loss_G_ID * self._lambda_id \
                    + feat_match_loss * self._lambda_feat

                if self._current_step % 2 == 0:
                    loss_G_Rec = self._model.criterionRec(
                        img_fake,
                        img_1,
                    ) * self._conf.model_config.options['lambda_rec']
                    self.update_meter('loss_G_Rec', loss_G_Rec.item())
                    loss_G += loss_G_Rec

                self._optim_g.zero_grad()
                loss_G.backward()
                self._optim_g.step()

        if (self._current_step + 1) % self._sample_freq == 0:
            self._model.netG.eval()
            bs = self._conf.batch_size
            with torch.no_grad():
                imgs = [torch.zeros_like(img_1[0]).cpu()]
                save_img = img_1.cpu() * self._imagenet_std + \
                    self._imagenet_mean
                imgs.extend([img for img in save_img])

                id_vector_src1 = self._make_latent_id(img_2)

                for i in range(bs):
                    imgs.append(save_img[i])
                    img_fake = self._model.netG(
                        img_1[i].repeat(bs, 1, 1, 1),
                        id_vector_src1,
                    ).cpu()
                    img_fake = img_fake * self._imagenet_std + \
                        self._imagenet_mean
                    imgs.extend([img for img in img_fake])

                self.plot_samples(imgs)

    def plot_samples(self, samples: List[torch.Tensor]) -> None:
        image_grid = torchvision.utils.make_grid(
            samples,
            nrow=self._conf.batch_size + 1,
            padding=0,
        )
        path = self._sample_path / f'step_{self._current_step + 1}.jpg'
        torchvision.utils.save_image(image_grid, path, nrow=1)
        if self._conf.wandb:
            wandb.log({path.stem: [wandb.Image(str(path))]})