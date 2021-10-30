import numpy as np 
import pandas as pd
import os
import time 
from tqdm.auto import tqdm 
import wandb 
from kaggle_secrets import UserSecretsClient 
import gc 

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score 

import torch 
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset 

from dataset import SartoriusTrainDataset

class Trainer:
    def __init__(self, df, config, model):
        self.df = df 
        self.best_threshold = None 
        self.config = config 
        self.model = model 
        self.main()
        
    def main(self):
        threshold_list = []
        kf = GroupKFold(n_splits=2 if self.config["debug"] else self.config["n_fold"])
        for fold, (tr, va) in enumerate(kf.split(self.df, self.df.annotation, self.df.id)):
            print("="*40, f"fold: {fold+1}", "="*40)
            ds_train = SartoriusTrainDataset(self.df.iloc[tr])
            ds_val = SartoriusTrainDataset(self.df.iloc[va], is_train=False)
            train_dl = DataLoader(ds_train, batch_size=self.config["batch_size"], shuffle=True, drop_last=True, 
                                 pin_memory=True, num_workers=3)
            val_dl = DataLoader(ds_val, batch_size=self.config["batch_size"], shuffle=False, drop_last=False, 
                                 pin_memory=True, num_workers=3)
            
            model = self.fit(train_dl, val_dl, fold)
            threshold = self.predict(val_dl, model, fold)
            threshold_list.append(threshold)
        self.best_threshold = np.mean(np.array(threshold_list))
        del ds_train, ds_val, train_dl, val_dl, model 
        gc.collect()
            
        
    def fit(self, train, val, fold: int):
        model = self.model 
        model.to(self.config["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        criterion = nn.BCEWithLogitsLoss()
        run = self.setup_db(fold)
        wandb.watch(model)
        
        best_model, best_loss, best_e = None, np.inf, None
        loss_tr, loss_va = [], []
        for e in range(1 if self.config["debug"] else self.config["epoch"]):
            ts = time.time()
            tr_loss = self.train_f(train, model, criterion, optimizer)
            va_loss = self.train_f(val, model, criterion, None, False)
            now = time.time()
            loss_tr.append(tr_loss)
            loss_va.append(va_loss)
            print(
                f"fold: {fold+1} | epoch: {e+1} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f} | dilation: {round((now-ts)/60.0, 2)}s"
            )
            
            if best_loss > va_loss:
                best_model = model 
                best_loss = va_loss 
                best_e = e 
            gc.collect()
        wandb.log({
            "fold": fold, 
            "best_loss": best_loss,
            "best_epoch": best_e,
            "train_loss": loss_tr, 
            "val_loss": loss_va
        })
        print(f"best val loss: {best_loss:.4f}")
        self.checkpoint(best_model, fold+1, best_loss, best_e)
        return best_model
    
    def predict(self, dl, model, fold):
        model.eval()
        with torch.no_grad():
            for d in tqdm(dl):
                x = d["image"].to(self.config["device"])
                t = d["mask"].to(self.config["device"])
                
                y = model(x)
                y = torch.sigmoid(y)
                del x, t 
                break
            self.save_imgs(y, fold)
            
            predv, corrv, acc_list, threshold = [], [], [], []
            for d in tqdm(dl):
                x = d["image"].to(self.config["device"])
                t = d["mask"].to(self.config["device"])
                
                y = model(x)
                y = torch.sigmoid(y)
                for yy in y[:, 0, :, :].detach().cpu().numpy():
                    predv.append(yy.flatten())
                for tt in t[:, 0, :, :].detach().cpu().numpy():
                    corrv.append(tt.flatten())
                    
            predv, corrv = np.array(predv), np.array(corrv).astype(np.uint8)
            for th in np.arange(0.35, 0.95, 0.05):
                predv_ = np.where(predv >= th, 1, 0).astype(np.uint8)
                acc = accuracy_score(predv_.flatten(), corrv.flatten())
                acc_list.append(acc)
                threshold.append(th)
            indics = np.argmax(np.array(acc_list))
            print(f"fold: {fold+1} | best score: {acc_list[indics]} | best threshold: {threshold[indics]}")
            del predv, corrv 
        return threshold[indics]
                
                
    
    def train_f(self, dl, model, criterion, optimizer, is_train=True):
        total_loss = []
        if is_train:
            model.train()
        else:
            model.eval()
            
        for d in tqdm(dl):
            x = d["image"].to(self.config["device"])
            t = d["mask"].to(self.config["device"])
            
            if is_train:
                y = model(x)
                loss = criterion(y, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            else:
                with torch.no_grad():
                    y = model(x)
                    loss = criterion(y, t)
            total_loss.append(loss.item())
            del x, t, loss 
        return np.mean(np.array(total_loss))
    
    def checkpoint(self, model, fold, loss, e):
        os.makedirs("/kaggle/working/models", exist_ok=True)
        torch.save(
            model.state_dict(), 
            f"/kaggle/working/models/sartorius_fold{fold}_epoch{str(e)}_loss{str(round(loss, 3))}.pth"
        )
        
    def save_imgs(self, y, fold):
        os.makedirs("/kaggle/working/generated", exist_ok=True)
        save_image(y, f"/kaggle/working/generated/{str(fold+1)}.jpg", nrow=y.size()[0]//2)

    def setup_db(self, fold):
        if fold == 0:
            user_secrets = UserSecretsClient()
            api = user_secrets.get_secret("wandb_api") 
            wandb.login(key=api)
        run = wandb.init(
            project = self.config["competition"], 
            name = self.config["model_name"], 
            config = self.config, 
            group = self.config["model_name"], 
            job_type = self.config["type"]
        )
        return run
