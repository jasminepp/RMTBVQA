import torch
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from modules.RMViTModel import RMViTModel
from torchvision import transforms
import numpy as np

import os
import argparse
import pickle
import skvideo.io
from PIL import Image
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class torch_transform:
    def __init__(self, size):
        self.transform1 = transforms.Compose(
            [
                transforms.Resize((size[0],size[1])),
                transforms.ToTensor(),
            ]
        )
        
        self.transform2 = transforms.Compose(
            [
                transforms.Resize((size[0] // 2, size[1] // 2)),
                transforms.ToTensor(),
            ]
        )
    
    def __call__(self, x):
        return self.transform1(x), self.transform2(x)


def extract_features_temporal(args, model, batch_im, batch_im_2):
    feat = []
    
    model.eval()
   
    batch_im = batch_im.type(torch.float32)
    batch_im_2 = batch_im_2.type(torch.float32)
    
    batch_im = batch_im.cuda(non_blocking=True).unsqueeze(0)
    batch_im_2 = batch_im_2.cuda(non_blocking=True).unsqueeze(0)
    
    with torch.no_grad():
        _, _, model_feat, model_feat_2 = model(batch_im, batch_im_2)
    
    feat_ = np.hstack((model_feat.detach().cpu().numpy(),\
                            model_feat_2.detach().cpu().numpy()))
    feat.extend(feat_)
    return np.array(feat)

def main(args):
    temporal_model = RMViTModel(num_mem_token=12,emb_dim =2048, segment_size=4, projection_dim=args.projection_dim, normalize= args.normalize)
    temporal_model.load_state_dict(torch.load(args.temporal_model_path, \
                                              map_location=args.device.type))
    temporal_model = temporal_model.to(args.device)
    feature_folder = args.feature_folder
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    fls = pd.read_csv(args.csv_file)
    fls = fls.loc[:,'FilePath'].tolist()
    
    for feat_name in fls:
        feat_path = feature_folder + feat_name + '/' + feat_name +'.npy'
        video_feat = np.load(feat_path, allow_pickle = True)
        feat_frames = torch.from_numpy(video_feat[:,:2048])
        feat_frames_2 = torch.from_numpy(video_feat[:,2048:])
        video_feat = extract_features_temporal(args, temporal_model, feat_frames, feat_frames_2)
        feature_save_path = args.feature_save_folder + feat_name +'.npy'
        feature_save_dir = os.path.dirname(feature_save_path)
        if not os.path.exists(feature_save_dir):
            os.makedirs(feature_save_dir)
        np.save(feature_save_path, video_feat)
        print(f"{feature_save_path} Done")
        
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video_path', type=str, \
                        default='sample_videos/30.mp4', \
                        help='Path to video', metavar='')
    parser.add_argument('--spatial_model_path', type=str, \
                        default='./models/spatial_feature_extractor.tar', \
                        help='Path to trained spatial_feature_extractor', metavar='')
    parser.add_argument('--temporal_model_path', type=str, \
                        default='./checkpoints/', \
                        help='Path to trained RMT-BVQA model', metavar='')
    parser.add_argument('--num_frames', type=int, \
                        default=16, \
                        help='number of frames fed to RMT', metavar='')
    parser.add_argument('--feature_save_folder', type=str, \
                        default='./test/features/', \
                        help='path to save features', metavar='')
    parser.add_argument('--projection_dim', type = int, default = 128,\
                        help = 'dimensions of the output feature from projector')
    parser.add_argument('--normalize', type = bool, default = True,\
                        help = 'normalize encoder output')
    parser.add_argument('--csv_file', type=str, \
                        default='./csv_files/frames_feature.csv', \
                        help='path for csv file with filenames', metavar='')
    parser.add_argument('--feature_folder', type=str, \
                        default='./test/VDPVE_features/', \
                        help='write folder', metavar='')
    
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)