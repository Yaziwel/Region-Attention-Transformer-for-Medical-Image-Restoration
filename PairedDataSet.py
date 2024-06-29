from utils import load_img, save_img, load_itk, save_itk, tensor2img, img2tensor 
import torch 
from torch.utils.data import Dataset 
import os 
import glob 
import random


class PairedData(Dataset): 
    def __init__(self, root, target = 'train', use_fine_mask = True, use_coarse_mask=False, use_num=-100): 
        super(Dataset, self).__init__() 
        self.use_fine_mask = use_fine_mask
        self.use_coarse_mask = use_coarse_mask
        
        name_list = os.listdir(os.path.join(root, target, "fd")) 
        
        self.HR_path = []
        self.LR_path = [] 
        self.MASK_fine = []
        self.MASK_coarse = [] 
        
        for i, name in enumerate(name_list):
            self.HR_path.append(os.path.join(root, target, "fd", name)) 
            self.LR_path.append(os.path.join(root, target, "qd", name)) 
            self.MASK_fine.append(os.path.join(root, target, "qd_seg_123456", name.replace('.jpg', '.nii.gz').replace('.png', '.nii.gz').replace('.bmp', '.nii.gz'))) 
            self.MASK_coarse.append(os.path.join(root, target, "qd_seg_123", name.replace('.jpg', '.nii.gz').replace('.png', '.nii.gz').replace('.bmp', '.nii.gz'))) 
            if i==use_num-1:
                break
            
        self.length = len(self.HR_path) 
        self.target = target
        
        
    
    def __len__(self):
        return self.length 
    


    def __getitem__(self, idx): 
        

        

        hr_img = img2tensor(load_img(self.HR_path[idx], grayscale=True))
        lr_img = img2tensor(load_img(self.LR_path[idx], grayscale=True)) 

        
        
        _, file_name = os.path.split(self.LR_path[idx])
        
        if self.use_fine_mask:
            mask_fine = torch.from_numpy(load_itk(self.MASK_fine[idx])) 
            
        else:
            mask_fine = []
        if self.use_coarse_mask:
            mask_coarse = torch.from_numpy(load_itk(self.MASK_coarse[idx])) 
            
        else:
            mask_coarse = [] 
            

        if self.target == "train":
            i = random.choice([1, 2, 3, 4]) 
            hr_img = torch.rot90(hr_img, i, [1, 2]) 
            lr_img = torch.rot90(lr_img, i, [1, 2]) 
            if self.use_fine_mask:
                mask_fine = torch.rot90(mask_fine, i, [0, 1]) 
            if self.use_coarse_mask:
                mask_coarse = torch.rot90(mask_coarse, i, [0, 1]) 
        # import pdb 
        # pdb.set_trace()
            
        return hr_img, lr_img, mask_fine, mask_coarse, file_name
    