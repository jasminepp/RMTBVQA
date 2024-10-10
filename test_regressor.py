import numpy as np
import pandas as pd
import pickle
import os
def test_regressor(test_csv_path, regressor_path, output_path,feature_save_folder):

    with open(regressor_path, 'rb') as f:
        reg = pickle.load(f)
    

    test_data = pd.read_csv(test_csv_path)
    
  
    predicted_scores = []
    for index, row in test_data.iterrows():
        feat_path = feature_save_folder + row[0]  
        if not os.path.exists(feat_path):
           continue
        feat = np.load(feat_path)
        score = np.mean(reg.predict(feat))
        predicted_scores.append(score)
    

    with open(output_path, 'w') as f:
        for score in predicted_scores:
            f.write(f"{score}\n")

if __name__ == "__main__":
    test_csv_path = "./csv_files/regression_test.csv"  
    regressor_path = "./regressor/regressor.save"  
    output_path = "results/result.txt"  
    feature_save_folder = "./test/features/"
    
    test_regressor(test_csv_path, regressor_path, output_path,feature_save_folder)