import argparse
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import csv
from PIL import Image
import numpy as np
import pretrainedmodels
import timm
from Resnet import resnet101_denoise, resnet152_denoise
from importlib import import_module

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

class AdvData(Dataset):
    def __init__(self,img_dir,csv_file,transform=None):
        super(AdvData, self).__init__()
        self.imgs = np.load(img_dir)
        self.csv_file = csv_file
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.csv_file,'r',encoding='utf=8'))
        next(reader)
        self.selected_dict = list(reader)
    def __getitem__(self, index):
        label = int(self.selected_dict[index][6])-1
        target = int(self.selected_dict[index][7])-1
        image = Image.fromarray(np.uint8(self.imgs[index].transpose(1,2,0)))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, target        
    def __len__(self):
        return len(self.selected_dict)

parser = argparse.ArgumentParser(description='Ghost Sample Attack')
parser.add_argument('--exp', '-e', metavar='MODEL', default='sample',help='model')
parser.add_argument('--img_dir', type=str, default='./result/resnet50_16_MaxLogit_10_GSA.npy')
args = parser.parse_args()
print('Hyper-parameters: {}\n'.format(args.__dict__))
img_dir = args.img_dir
csv_file = './data/images.csv'
epsilon = 16/255
batch_size = 20
target_models=['resnet50','densenet121','vgg16_bn','inception_v3','mobilenet_v2','SENet','PNASNet']
# target_models=['adv_inception_v3','ens_adv_inception_resnet_v2','Resnext101-DenoiseAll','Fast_AT','DeepAugment_AugMix','HGD']
# target_models=['pit_s_224', 'cait_s24_224', 'deit_base_patch16_224']
targeted = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True

# t = timm.list_models(pretrained=True)
# print(t)
for target_model in target_models:
    input_size = [3,224,224]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if(target_model == 'inception_v3' or target_model == 'HGD' or 'adv' in target_model):
        input_size = [3, 299, 299]
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif(target_model == 'PNASNet'):
        input_size = [3, 331, 331]
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif('Denoise' in target_model):
        input_size = [3, 224, 224]
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
    trans = transforms.Compose([
        transforms.Resize(tuple((input_size[1:]))),
        transforms.ToTensor()
    ])
    norm = transforms.Normalize(tuple(mean),tuple(std))

    dataset = AdvData(img_dir,csv_file,transform=trans)
    data_loader = DataLoader(dataset,batch_size = batch_size,shuffle=False, num_workers = 8, pin_memory = False)
    
    if target_model == 'PNASNet':
        model = timm.create_model('pnasnet5large',pretrained=True)
        model.eval()
    elif target_model == 'SENet':
        model =  pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
        # model = timm.create_model('seresnet50',pretrained=True)
        model.eval()
    elif 'adv' in target_model or 'inception' in target_model:
        model = timm.create_model(target_model,pretrained=True)
        model.eval()
    elif target_model == 'pit_s_224' or target_model =='cait_s24_224'or target_model =='deit_base_patch16_224'or target_model =='swin_base_patch4_window7_224':
        model = timm.create_model(target_model,pretrained=True)
        model.eval()
    elif target_model =='HGD':
        modelpath = os.path.join(os.path.abspath('./Exps'),args.exp)
        sys.path.append(modelpath)
        m = import_module('model')
        config, model = m.get_model()
        checkpoint = torch.load('./models/denoise_incepv3_012.ckpt')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.cuda()
        model.eval()
    elif target_model == 'AdvResnet152':
        model = torchvision.models.resnet152(pretrained=False)
        model.load_state_dict(torch.load("models/ckpts/res152-adv.checkpoint"))
    elif target_model == 'Resnext101-DenoiseAll':
        model = resnet101_denoise()
        model.load_state_dict(torch.load("models/ckpts/Adv_Denoise_Resnext101.pytorch"))
    elif target_model == 'Resnet152-DenoiseAll':
        model = resnet152_denoise()
        model.load_state_dict(torch.load("models/ckpts/Adv_Denoise_Resnet152.pytorch"))
    elif target_model == "DeepAugment_AugMix":
        # The many faces of robustness: A critical analysis of out-of-distribution generalization, ICCV 2021
        # https://github.com/hendrycks/imagenet-r
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load("models/ckpts/deepaugment_and_augmix.pth.tar", map_location='cpu')
        model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items()})
    elif target_model == "Fast_AT":
        # Fast adversarial training using FGSM, ICLR 2020
        # https://github.com/locuslab/fast_adversarial/tree/master/ImageNet
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load("models/ckpts/imagenet_model_weights_4px.pth.tar", map_location='cpu') 
        model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items()})
    else:
        model = models.__dict__[target_model](pretrained=True).eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    asr=0.0
    predict = []
    for i, (img,label,target) in enumerate(data_loader):
        img,label,target = img.to(device),label.to(device),target.to(device)

        with torch.no_grad():
            output = model(norm(img))
        if targeted:
            asr += torch.sum(output.argmax(dim=-1) == target).item()
        else:
            asr += torch.sum(output.argmax(dim=-1) != label).item()
        del img

    print('{} Blackbox Attack Success Rate:{}\n'.format(target_model,asr/len(dataset)))
    del model