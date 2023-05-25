
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF','npy','mat'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
from copy import deepcopy

class ImageDataset(data.Dataset):
    def __init__(self, opts):
        self.img_paths = sorted(make_dataset(opts.data_root))
        self.is_train=not opts.eval
        input_size=opts.input_size
        output_size=opts.output_size
        per_edge_pad=(output_size-input_size)//2
        image_mean=opts.image_mean
        image_std=opts.image_std

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(output_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
            ])

        self.input_image_normalize=transforms.Normalize(mean=torch.tensor([image_mean,image_mean,image_mean]),  std=torch.tensor([image_std,image_std,image_std]))

        self.mask=torch.zeros([1,output_size, output_size])

        self.mask[:,per_edge_pad:-per_edge_pad,per_edge_pad:-per_edge_pad]=1

        self.per_edge_pad=per_edge_pad

    def __getitem__(self, index):
        name= os.path.splitext(os.path.split(self.img_paths[index])[-1])[0]
        im=Image.open(self.img_paths[index]).convert('RGB')
        im=self.transform(im)
        input_img=self.input_image_normalize(deepcopy(im))[:,self.per_edge_pad:-self.per_edge_pad,self.per_edge_pad:-self.per_edge_pad]

        gt=self.input_image_normalize(deepcopy(im))
        gt_inner=deepcopy(gt)*self.mask
        return {'input':input_img,'ground_truth':gt,'gt_inner':gt_inner,'name':name}

    def __len__(self):
        return len(self.img_paths)
