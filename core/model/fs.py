import torch
import torch.nn as nn

from core.model.fs_networks import GeneratorAdainUpsample
from core.model.pg_modules.projected_discriminator import \
    ProjectedDiscriminator


class FS(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def initialize(self, opt):
        self._is_train = opt['isTrain']

        # Generator network
        self.netG = GeneratorAdainUpsample(
            input_nc=3,
            output_nc=3,
            latent_size=512,
            n_blocks=9,
            deep=opt['Gdeep'],
        )
        self.netG.cuda()

        netArc_checkpoint = torch.load(
            opt['arc_path'],
            map_location=torch.device('cpu'),
        )
        self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        # TODO try this
        if not self._is_train:
            pretrained_path = opt['checkpoints_dir']
            self.load_network(
                self.netG,
                'G',
                opt['which_epoch'],
                pretrained_path)
            return
        self.netD = ProjectedDiscriminator()
        self.netD.cuda()

        if self._is_train:
            self.criterionFeat = nn.L1Loss()
            self.criterionRec = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt['lr'],
                betas=(opt['beta1'], 0.99),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt['lr'],
                betas=(opt['beta1'], 0.99),
            )

        torch.cuda.empty_cache()
