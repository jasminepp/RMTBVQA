from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np
import pandas as pd

class video_data_feat(Dataset):
    def __init__(self, file_path):
        self.fls = pd.read_csv(file_path)
        self.tranform_toT = transforms.Compose([
                transforms.ToTensor(),
                ])
    
    def __len__(self):
        return len(self.fls)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        vid_path = self.fls.iloc[idx]['FilePath']
        base_path = './training_data/features/'
        name = vid_path
        div_factor1 = np.random.choice([1,2],1)[0]
        colorspace_choice1 = np.random.choice([0,1,2,3,4],1)[0]
        temporal_choice1 = np.random.choice([0,1,2,3],1)[0]
        
        feat_path1 = base_path+ name + '/' \
                + name + '_color'+str(colorspace_choice1) +'_temp' + str(temporal_choice1) + '.npy'
        feat1 = np.load(feat_path1, allow_pickle = True)

      
        T,D = feat1.shape
        if T < 32:
            padding = 32 - T
            feat1= np.pad(feat1, ((padding, 0), (0, 0)), 'constant', constant_values=0)
        elif T > 32:
  
            start_index = np.random.randint(0, T - 32 + 1)
       
            feat1 = feat1[start_index:start_index + 32, :]
       
        #choose the scale
        if div_factor1 == 1:
            feat1 = feat1[:,:D // 2]
        else:
            feat1 = feat1[:,D//2:]
        
        feat1 = self.tranform_toT(feat1)
        
        # determine second video characteristics
        div_factor2 = 3 - div_factor1
        colorspace_choice2 = np.random.choice([0,1,2,3,4],1)[0]
        temporal_choice2 = np.random.choice([0,1,2,3],1)[0]
       
        feat_path2 = base_path + name + '/' + name + '_color'+ str(colorspace_choice2) +'_temp' + str(temporal_choice2) + '.npy'
        feat2 = np.load(feat_path2, allow_pickle=True)
        T,D = feat2.shape
        if T < 32:
            padding = 32 - T
            feat2 = np.pad(feat2, ((padding, 0), (0, 0)), 'constant', constant_values=0)
        elif T > 32:
            start_index = np.random.randint(0, T - 32 + 1)
            feat2 = feat2[start_index:start_index + 32, :]
        
        
        #choose the scale
        if div_factor2 == 1:
            feat2 = feat2[:,:D // 2]
        else:
            feat2 = feat2[:,D//2:]
        
        feat2 = self.tranform_toT(feat2)

        
        quality_label = self.fls.iloc[idx]['quality_label']
        quality_label = quality_label[1:-1].split(' ')
        quality_label = np.array([t.replace(',','') for t in quality_label]).astype(np.float16)
        
        content_label = self.fls.iloc[idx]['content_label'] 
        content_label = np.array(content_label[1:-1].split(',')).astype(np.float32)
        
        return feat1, feat2, quality_label, content_label