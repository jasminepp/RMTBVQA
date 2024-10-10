
import numpy as np
import argparse
from sklearn.linear_model import Ridge
import pickle
import pandas as pd  


def main(args):
    data_path = "./csv_files/regression_train.csv"
    data = pd.read_csv(data_path)
    
    all_feats = []
    all_scores = []

    for index, row in data.iterrows():
        feat_path = args.feature_folder + row[0] 
        score = row[2]  
        
        feat = np.load(feat_path)
        feat_mean = np.mean(feat, axis=0)
        all_feats.append(feat_mean)
        all_scores.append(score)
    

    all_feats = np.array(all_feats)
    print(all_feats.shape)
    all_scores = np.array(all_scores)
    
    reg = Ridge(alpha=args.alpha).fit(all_feats, all_scores)
    pickle.dump(reg, open('./regressor/regressor.save', 'wb'))

def parse_args():
    parser = argparse.ArgumentParser(description="linear regressor")
    parser.add_argument('--alpha', type=float, default=0.1, help='regularization coefficient')
    parser.add_argument('--feature_folder', type=str, \
                        default='./test/features/', \
                        help='path to save features', metavar='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)