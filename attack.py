import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import time
from tqdm import tqdm
import csv
from PIL import Image
import numpy as np
import argparse
import scipy.stats as st
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Ghost Sample Attack')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--epsilon', type=int, default=16)
parser.add_argument('--alpha', type=float, default=2/255)
parser.add_argument('--src_model', type=str, default='resnet50', choices=['resnet50','vgg16_bn','inception_v3'])
parser.add_argument('--loss_function', type=str, default='MaxLogit', choices=['CE','MaxLogit'])
parser.add_argument('--img_dir', type=str, default='./data/images/')
parser.add_argument('--label_dir', type=str, default='./data/images.csv')
parser.add_argument('--save_dir', type=str, default='./temp/',help='directory stores temporary batch data')

parser.add_argument('--targeted', action='store_true')
parser.add_argument('--MI', action='store_true')
parser.add_argument('--mu', type=float, default=1.0, help='momentum factor')
parser.add_argument('--DI', action='store_true')
parser.add_argument('--TI', action='store_true')
parser.add_argument('--SI_num', type=int, default=0)
parser.add_argument('--NI', action='store_true')

parser.add_argument('--Admix_param', type=float, default=0)
parser.add_argument('--m1', type=int, default=3, help='number of randomly sampled images')
parser.add_argument('--m2', type=int, default=5, help='num of copies')

# GSA parameters
parser.add_argument('--GSA', action='store_true',default=True)
parser.add_argument('--aug_num', type=int, default=15)
parser.add_argument('--sigma', type=float, default=0.1)

parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()
print('Hyper-parameters: {}\n'.format(args.__dict__))

s=1234
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

class SelectedData(Dataset):
    def __init__(self,img_dir,csv_file,transform=None):
        super(SelectedData, self).__init__()
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.csv_file,'r',encoding='utf=8'))
        next(reader)
        self.selected_dict = list(reader)
    def __getitem__(self, index):
        image_id = self.selected_dict[index][0]
        label = int(self.selected_dict[index][6])-1
        target = int(self.selected_dict[index][7])-1
        image = Image.open(os.path.join(self.img_dir,image_id+'.png'))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, target
     
    def __len__(self):
        return len(self.selected_dict)

def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.5:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def TI(kernel_size,channels):
    kernel = gkern(kernel_size, channels).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    return gaussian_kernel

def SI(args,input_tensor):
    img = input_tensor.clone()
    for p in range(args.SI_num):
        if p==0:
            img = input_tensor/(2**p)
        else:
            img = torch.cat((img,input_tensor/(2**p)),dim=0)
    return img

def admix(args,input_tensor):
    img = input_tensor.clone()
    size = input_tensor.shape[0]
    index_list = [n for n in range(size)]
    for p in range(args.m1):
        # Shuffle input_tensor w.r.t dim 0
        img_mix = input_tensor.cpu().detach().numpy()
        np.random.shuffle(index_list)     
        shuff_img=[]
        for k in index_list:
            shuff_x=img_mix[k,:,:]
            shuff_img.append(np.expand_dims(shuff_x,axis=0))
        img_mix=torch.from_numpy(np.concatenate(shuff_img,axis=0)).to(device)
        for q in range(args.m2):   
            # lam = np.random.beta(1,1)
            if(p==0 and q==0):
                img = (input_tensor+args.Admix_param*img_mix)/(2**q)
            else:
                img = torch.cat((img,(input_tensor+args.Admix_param*img_mix)/(2**q)),dim=0)
    del img_mix
    return img

def gaussian_aug(args, input_tensor):
    batch = input_tensor.repeat((args.aug_num-1, 1, 1, 1))
    noise = torch.randn_like(batch, device='cuda') * args.sigma
    batch = torch.cat((input_tensor,batch+noise),dim=0)
    return batch

def duplicate(input_tensor,n):
    list = []
    for l in range(n):
        list.append(input_tensor)
    input_tensor = torch.cat(list,dim=0)
    return input_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = True

input_size = [3,224, 224]
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
if(args.src_model == 'inception_v3'):
    input_size = [3, 299, 299]
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
trans = transforms.Compose([
    transforms.Resize(tuple((input_size[1:]))),
    transforms.ToTensor()
])
norm = transforms.Normalize(tuple(mean),tuple(std))

# Load data
dataset = SelectedData(args.img_dir,args.label_dir,transform=trans)
data_loader = DataLoader(dataset,batch_size = args.batch_size,shuffle=False, num_workers = 8, pin_memory = False)

# Load model
model = models.__dict__[args.src_model](pretrained=True).eval()
model.to(device)

criterion = nn.CrossEntropyLoss()
asr=0.0

for i, (img,label,target) in enumerate(tqdm(data_loader)):
    img,label,target = img.to(device),label.to(device),target.to(device)
    ori_img, ori_label, ori_target = img.clone(), label.clone(), target.clone()
    model.zero_grad()
    grad = 0.0
    predicts = []
    for j in (range(args.iters)):
        if args.GSA: # GSA
            img_tmp = gaussian_aug(args,img)
            label = duplicate(label,args.aug_num).to(device)
            if args.targeted:
                target = duplicate(target,args.aug_num).to(device)
        elif args.Admix_param > 0: # Admix
            img_tmp = admix(args, img)
            label = duplicate(label,args.m1*args.m2).to(device)
            if args.targeted:
                target = duplicate(target,args.m1*args.m2).to(device)
        elif args.SI_num > 0:
            if args.NI:
                img_tmp = SI(args,img+args.mu*args.alpha*grad)
            else:
                img_tmp = SI(args,img)
            label = duplicate(label,args.SI_num).to(device)
            if args.targeted:
                target = duplicate(target,args.SI_num).to(device)
        else:
            img_tmp = img
        img_tmp.requires_grad_(True)

        if args.DI:  # DI
            grad = 0
            for c in range(args.aug_num):
                logits = model(norm(DI(img_tmp)))
                loss_func = nn.CrossEntropyLoss(reduction='sum')
                loss = -1 * loss_func(logits, label)
                loss.backward()
                grad = grad + img_tmp.grad.clone() 
        else:
            logits = model(norm(img_tmp))
        
        if args.loss_function == 'CE':
            loss_func = nn.CrossEntropyLoss(reduction='sum')
            if args.targeted:
                loss = loss_func(logits, target)
            else:
                loss = -1 * loss_func(logits, label)
        elif args.loss_function == 'MaxLogit':
            if args.targeted:
                real = logits.gather(1,target.unsqueeze(1)).squeeze(1)
                loss = -1 * real.sum()
            else:
                real = logits.gather(1,label.unsqueeze(1)).squeeze(1)
                loss = real.sum()
        
        loss.backward()
        
        if args.GSA:
            chunk_num = args.aug_num
            cur_grad = 0
            grads = img_tmp.grad.data.chunk(args.aug_num,dim=0)
            for g in grads:
                cur_grad += g/(args.aug_num)
            label = label.chunk(args.aug_num,dim=0)[0]
            if args.targeted:
                target = target.chunk(args.aug_num,dim=0)[0]
        elif args.Admix_param > 0:
            cur_grad = 0
            grads = img_tmp.grad.data.chunk(args.m1*args.m2,dim=0)
            for g in grads:
                cur_grad += g/(args.m1*args.m2)
            label = label.chunk(args.m1*args.m2,dim=0)[0]
            if args.targeted:
                target = target.chunk(args.m1*args.m2,dim=0)[0]
        elif args.SI_num > 0:
            cur_grad = 0
            grads = img_tmp.grad.data.chunk(args.SI_num,dim=0)
            for g in grads:
                cur_grad += g/(args.SI_num)
            label = label.chunk(args.SI_num,dim=0)[0]
            if args.targeted:
                target = target.chunk(args.SI_num,dim=0)[0]
        elif args.DI:
            chunk_num = args.aug_num
            cur_grad = 0
            grads = img_tmp.grad.data.chunk(args.aug_num,dim=0)
            for g in grads:
                cur_grad += g/(args.aug_num)
            label = label.chunk(args.aug_num,dim=0)[0]
            if args.targeted:
                target = target.chunk(args.aug_num,dim=0)[0]
        else:
            cur_grad= img.grad.data
                
        if args.TI:  # TI
            gaussian_kernel = TI(5,3)
            cur_grad = F.conv2d(cur_grad, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
        
        if args.MI:  # MI
            grad = args.mu*grad + cur_grad
        else:
            grad = cur_grad
    
        img = img.data - args.alpha * torch.sign(grad)
        
        img = torch.clamp(img, ori_img - args.epsilon/255, ori_img + args.epsilon/255)
        img = torch.clamp(img, min=0, max=1)
       
    with torch.no_grad():
        output = model(norm(img))
    if args.targeted:
        asr += torch.sum(output.argmax(dim=-1) == ori_target).item()
    else:
        asr += torch.sum(output.argmax(dim=-1) != ori_label).item()
    
    np.save(args.save_dir+'/batch_{}.npy'.format(i), torch.round(img.data*255).cpu().numpy().astype(np.uint8()))
    print('batch_{}.npy saved, Whitebox Attack Success Count:{}'.format(i,asr))
    del img, img_tmp, label, target, ori_img, ori_label, ori_target, logits, grad

print('{} Whitebox Attack Success Rate:{}'.format(args.src_model,asr/len(dataset)))

exp_name = args.src_model+'_'+str(args.epsilon)+'_'+args.loss_function+'_'+ str(args.iters)
if args.MI:
    exp_name += '_MI'
if args.DI:
    exp_name += '_DI'
if args.TI:
    exp_name += '_TI'
if args.SI_num > 0:
    exp_name += '_SI'+ str(args.SI_num)
if args.Admix_param > 0:
    exp_name += '_Admix'
if args.GSA:
    exp_name += '_GSA'
if args.targeted:
    exp_name += '_targeted'

files = os.listdir(args.save_dir)
files.sort(key=lambda x:int(x[6:-4]))
data = np.zeros((3,224,224))
for i, file in enumerate(files):
    newdata = np.load(args.save_dir+file)
    if(i==0):
        data = newdata
    else:
        data = np.concatenate((data,newdata),axis=0)
np.save('./result/'+exp_name+'.npy',data)

print('Hyper-parameters: {}\n'.format(args.__dict__))
print(exp_name)