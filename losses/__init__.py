import torch.nn as nn
from .reconstruct import ReconLoss
from .lpips import LPIPS
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,IMAGENET_INCEPTION_MEAN,IMAGENET_INCEPTION_STD


class SetCriterion(nn.Module):
    def __init__(self,opts):
        super().__init__()
        self.recon_loss=ReconLoss()
        self.perceptual_loss = LPIPS().cuda()
        self.gen_weight_dict={'loss_g_recon':5, 'loss_g_adversarial':1, 'loss_g_perceptual':10}
        self.dis_weight_dict = {'loss_d_adversarial': 1}
        self.imagenet_normalize=transforms.Normalize( mean=torch.tensor(IMAGENET_INCEPTION_MEAN),  std=torch.tensor(IMAGENET_INCEPTION_STD))
        self.image_mean=opts.image_mean
        self.image_std=opts.image_std
        self.discriminator_weight=0.8

    def renorm(self,tensor):
        tensor = tensor * self.image_std + self.image_mean
        return self.imagenet_normalize(tensor)

    def get_dis_loss(self,  input_fake, input_real, discriminator=None):
        assert discriminator is not None
        return {'loss_d_adversarial': discriminator.calc_dis_loss(input_fake.detach(),input_real)}

    def get_gen_loss(self, input_fake, input_real,discriminator=None, warmup=False, last_layer=None):
        if not warmup:
            assert discriminator is not None and last_layer is not None
            g_loss_dict={'loss_g_adversarial': discriminator.calc_gen_loss(input_fake,input_real)}
            g_loss_dict['loss_g_recon']=self.recon_loss(input_fake,input_real)
            g_loss_dict['loss_g_perceptual']=self.perceptual_loss(self.renorm(input_fake),self.renorm(input_real)).mean()
            nll_loss=g_loss_dict['loss_g_recon']+g_loss_dict['loss_g_perceptual']
            g_loss=g_loss_dict['loss_g_adversarial']
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * self.discriminator_weight
            self.gen_weight_dict['loss_g_adversarial'] = d_weight
            return g_loss_dict
        else:
            return {'loss_g_recon':self.recon_loss(input_fake,input_real)}





