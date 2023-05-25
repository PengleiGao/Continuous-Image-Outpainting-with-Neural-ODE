import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import itertools
import datetime

torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from datasets import ImageDataset
from util.misc import cosine_scheduler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='build')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-4)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--eval', default=False, type=bool)
parser.add_argument('--half_precision', default=False, type=bool)

parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--output_size', type=int, default=192)
parser.add_argument('--data_root', type=str, default='data/train')

parser.add_argument('--EMBED_DIM', type=int, default=128)
parser.add_argument('--DEPTHS', type=list, default=[2, 2, 18, 2])
parser.add_argument('--NUM_HEADS', type=list, default=[4, 8, 16, 32])
parser.add_argument('--image_mean', type=float, default=0.5)
parser.add_argument('--image_std', type=float, default=0.5)


from models.swin_base import swin_former
#from models.cnn_base import cnn_net
from models.CNNDis import MsImageDis
from losses import SetCriterion

from engine import train_one_epoch

if __name__ == '__main__':
    opts = parser.parse_args()

    train_dataset = ImageDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opts.batch_size,
                                               num_workers=opts.num_workers, persistent_workers=opts.num_workers > 0,
                                               shuffle=True)

    #gen = cnn_net(opts=opts).cuda()
    gen = swin_former(opts=opts).cuda()
    cnn_dis = MsImageDis().cuda()


    g_param_dicts = [
        {"params": [p for n, p in gen.named_parameters() if
                    'conv_offset_mask' not in n and not 'transformer_encoder' in n], "lr_scale": 1},
        {"params": [p for n, p in gen.named_parameters() if 'conv_offset_mask' in n], "lr_scale": 0.1},
        {"params": [p for n, p in gen.named_parameters() if 'transformer_encoder' in n], "lr_scale": 1}
    ]

    opt_g = torch.optim.Adam(g_param_dicts, lr=opts.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    opt_d = torch.optim.Adam(itertools.chain(cnn_dis.parameters()), lr=opts.lr, betas=(0.0, 0.99), weight_decay=1e-4)
    lr_schedule_values = cosine_scheduler(opts.lr, opts.min_lr, opts.max_epoch, len(train_loader))

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + '_' + opts.name
    root = "./logs"
    logdir = os.path.join(root, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    visdir = os.path.join(logdir, "visuals")
    for d in [logdir, ckptdir, visdir]:
        os.makedirs(d, exist_ok=True)
    opts.visdir = visdir
    opts.ckptdir = ckptdir

    if opts.half_precision:
        g_grad_scaler = torch.cuda.amp.GradScaler()
    else:
        g_grad_scaler = None

    criterion = SetCriterion(opts)

    iteration = 1
    for epoch in range(opts.max_epoch):
        for i, param_group in enumerate(opt_g.param_groups):
            param_group["lr"] = opts.lr * param_group["lr_scale"]
        for i, param_group in enumerate(opt_d.param_groups):
            param_group["lr"] = opts.lr


        train_one_epoch(opts, gen, cnn_dis, criterion, train_loader, opt_g, opt_d, torch.device('cuda'), epoch,
                        g_grad_scale=g_grad_scaler)

        iteration += len(train_loader)

        torch.save({'gen': gen.state_dict()}, os.path.join(ckptdir, f'latest_gen.pth'))
        torch.save({'dis': cnn_dis.state_dict()}, os.path.join(ckptdir, f'latest_dis.pth'))
        if (epoch + 1) % 20 == 0 and epoch > 50:
            torch.save({'gen': gen.state_dict()}, os.path.join(ckptdir, f'{epoch + 1}.pth'))
            torch.save({'dis': cnn_dis.state_dict()}, os.path.join(ckptdir, f'{epoch + 1}_dis.pth'))
