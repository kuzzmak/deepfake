import torch
import torch.nn as nn

from core.model.fs_networks import GeneratorAdainUpsample
from core.model.pg_modules.blocks import IRBlock
from core.model.pg_modules.projected_discriminator import \
    ProjectedDiscriminator
from core.model.resnet import ResNet


class FS(nn.Module):

    NAME = 'fs'

    def __init__(self) -> None:
        super().__init__()

    def initialize(self, opt):
        self._train = opt['train']

        # Generator network
        self.netG = GeneratorAdainUpsample(
            input_nc=3,
            output_nc=3,
            latent_size=512,
            n_blocks=9,
            deep=opt['gdeep'],
        )
        self.netG.cuda()

        netArc_checkpoint = torch.load(
            opt['arc_path'],
            map_location=torch.device('cpu'),
        )
        self.netArc = ResNet(IRBlock, [3, 4, 23, 3])
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        self.netArc.load_state_dict(netArc_checkpoint['model'])
        self.netArc = self.netArc.cuda()
        # TODO try this
        if not self._train:
            pretrained_path = opt['checkpoints_dir']
            self.load_network(
                self.netG,
                'G',
                opt['which_epoch'],
                pretrained_path)
            return

        self.netD = ProjectedDiscriminator()
        self.netD.cuda()

        self.criterionFeat = nn.L1Loss()
        self.criterionRec = nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=opt['lr'],
            betas=opt['betas'],
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=opt['lr'],
            betas=opt['betas'],
        )

        torch.cuda.empty_cache()
