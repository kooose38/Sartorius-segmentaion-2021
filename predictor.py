import numpy as np 
import pandas as pd
import cv2 
import os
from tqdm.auto import tqdm 
from typing import List, Tuple 

import torch 
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset 

from dataset import SartoriusTestDataset

class Predictor:
    def __init__(self, threshold, config, model, model_path_name):
        self.threshold = threshold
        self.config = config 
        self.model = model
        self.model_path_name = model_path_name 
        self.main()
        
    def main(self):
        submission_list = []
        predt, image_id = self.fit()
        for p, id in zip(predt, image_id):
            # サイズの調整
            p = cv2.resize(p, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
            # 閾値によるピクセル変換
            p = self.post_process(p)
            for pp in p:
                submission_list.append((id, self.rle_encoding(pp.astype(int))))
        df = pd.DataFrame(submission_list, columns=["id", "predicted"])
        df.to_csv("/kaggle/working/submission.csv", index=False)
        
    def fit(self):
        '''
        Return:
            predt: [batch, weight, height]
            image_id: [str] * batch
        '''
        predt = []
        test_ds = SartoriusTestDataset()
        test_dl = DataLoader(test_ds, batch_size=self.config["batch_size"], shuffle=False, drop_last=False)
        for fold in range(2 if self.config["debug"] else self.config["n_fold"]):
            model = self.load_model(fold)
            pred, image_id = self.test_f(test_dl, model)
            predt.append(pred) 
        predt = np.mean(predt, 0)
        return predt, image_id
    
    def test_f(self, test_dl, model):
        with torch.no_grad():
            pred_list, ids_list = [], []
            for d in tqdm(test_dl):
                x = d["image"].to(self.config["device"])
                ids = d["image_id"]

                y = model(x)
                y = torch.sigmoid(y).detach().cpu().numpy()
                for yy in y:
                    pred_list.append(yy[0, :, :])
                for id in ids:
                    ids_list.append(id)
                del d
            return pred_list, ids_list 
        
    def load_model(self, fold):
        model = self.model
        if self.config["debug"] is not True:
            model_path = os.listdir(f"/kaggle/input/{self.model_path_name}")[fold]
            model.load_state_dict(torch.load(f"/kaggle/input/{self.model_path_name}/{model_path}", map_location={"cuda:0": "cpu"}))
            
        model.eval()
        return model.to(self.config["device"])
    
    def post_process(self, probability, min_size=300):
        '''
        Args:
            probability: [w, h] Type np.ndarray
            
        Return:
            prediction: [N, w, h] Type List
        '''
        mask = cv2.threshold(probability, self.threshold, 1, cv2.THRESH_BINARY)[1]
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = []
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                a_prediction = np.zeros((520, 704), np.float32)
                a_prediction[p] = 1
                predictions.append(a_prediction)
        return predictions
    
    def rle_encoding(self, x: List[float]) -> str:
        '''
        Args:
            x: [w, h] Type List

        '''
        dots = np.where(x.flatten() == 1)[0] # 1次元配列
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return ' '.join(map(str, run_lengths))