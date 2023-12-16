"""A trainer for ECG deep learning models."""

import os
import tqdm

from dataset import ECGDataset

import numpy as np
import pandas as pd
import torch
import sklearn


class Trainer:
    """
    A training object for ECG deep learning models.

    Arguments:
        cfg: The config dict, like config.py's cfg.
        device: A torch.device.
        model: A model, as defined in model.py.
        optim: A torch optimizer.
        scheduler: A torch scheduler.
        datasets: A dict of torch datasets for each split.
        dataloaders: A dict of torch dataloaders for each split.
        output: The output directory.

    Calling Trainer.train() will train a full model; calling Trainer.run_eval_on_split()
    and Trainer.run_eval_on_all() evaluates the model on a list of files and a directory
    of files.
    """
    def __init__(
        self,
        cfg,
        device,
        model,
        optim,
        scheduler,
        datasets,
        dataloaders,
        output):
        self.cfg = cfg
        self.device = device
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.output = output

        
    def train(self):
        n_epochs = self.cfg["optimizer"]["n_epochs"]
        
        # try to load former model
        epoch_resume, best_score, train_loss_history, train_score_histoty, valid_loss_history, valid_score_histoty = self.try_to_load()
        
        if (self.cfg["optimizer"]["reduce_on_plateau"] and 
            epoch_resume and
            self.cfg["optimizer"]["lr"] * self.cfg["optimizer"]["max_reduction"] 
                > self.optim.param_groups[0]["lr"]):
            epoch_resume = n_epochs        

        # run 
        for epoch in range(epoch_resume, n_epochs):
            for split in ["train", "valid"]:             
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_max_memory_allocated(i)
                    torch.cuda.reset_max_memory_cached(i)
                    
                # print(f"{split} - epoch {epoch} ")
                
                losses, ys, yhs, score = self.run_epoch(split)
                
                if split=="train":
                    train_loss_history.append(np.mean(losses))
                    train_score_histoty.append(score)
                    
                if split=="valid":
                    valid_loss_history.append(np.mean(losses))
                    valid_score_histoty.append(score)
                
                print(f"{split} {epoch} score: {score:.3f}", flush=True)
            
            # exam plateaued
            plateaued = False
            if self.cfg["optimizer"]["reduce_on_plateau"]:
                self.scheduler.step(np.mean(losses))
                # print("num bad epochs", self.scheduler.num_bad_epochs, flush=True)
                # print("lr", self.optim.param_groups[0]["lr"], flush=True)
                
                if (self.cfg["optimizer"]["lr"] * self.cfg["optimizer"]["max_reduction"]
                        > self.optim.param_groups[0]["lr"]):
                    print("plateaued", flush=True)
                    plateaued = True
            else:
                self.scheduler.step()
                
            best_score = self.save_model(losses, epoch, score, best_score, 
                                         train_loss_history, train_score_histoty, 
                                         valid_loss_history, valid_score_histoty)
            
            if plateaued:
                break

        best_epoch, best_score, _, _, _, _ = self.try_to_load("best.pt")
        print(f"Best score seen: {best_score:.3f} at epoch {best_epoch}", flush=True)
        
        self.run_eval_on_split("valid", report_performance=True)
        self.run_eval_on_split("test", report_performance=True)
        self.run_eval_on_split("all", report_performance=True)
        

    def run_epoch(self, split, no_label=False, dataloader=None):
        """
        calculate the 'losses, ys, yhs, score' in all items in dataloader
        """
        
        if not dataloader:
            # print(f"Loading dataloader for split {split}", flush=True)
            dataloader = self.dataloaders[split]

        if split == "train":
            self.model.train()
        else:
            self.model.eval()

        losses, ys, yhs = [], [], []
        
        for i, D in enumerate(tqdm.tqdm(dataloader)):
            if no_label:
                x = D[0].to(self.device)
                yh = self.model.module.forward(x)
                yhs.extend(yh.data.tolist())
            else:
                x = D[0].to(self.device)
                y = D[1].to(self.device)
                
                if self.device.type == "cuda":
                    (y, yh, loss) = self.model.module.train_step(x, y)
                else:
                    (y, yh, loss) = self.model.train_step(x, y)   
                
                assert not torch.isnan(x).any().item(), "x contain nan"
                assert not torch.isnan(yh).any().item(), "yh contain nan"
                assert not np.isnan(loss.item()), f"loss is nan at step {i}"
                
                if split=="train":
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    
                losses.append(loss.item())
                ys.extend(y.data.tolist())
                yhs.extend(yh.data.tolist())  

        ys, yhs, losses = np.array(ys), np.array(yhs), np.array(losses)

        if no_label:
            score = 0
            print('no label')
        else:
            if self.device.type == "cuda":
                score = self.model.module.score(ys, yhs, np.mean(losses))
            else:
                score = self.model.score(ys, yhs, np.mean(losses))
            
        return losses, ys, yhs, score 


    def run_eval_on_split(self, split, report_performance=False):
        
        print(f"Running eval on {split}")
        
        losses, ys, yhs, score = self.run_epoch(split)
        
        pd.DataFrame({"y": ys.flatten(),
                      "yh": yhs.flatten()}).to_csv(
            os.path.join(self.output, f"{split}_preds.csv"))
        
        print(f"{split} score: {score:.3f}")

        
    def run_eval_on_all(self):
        # using original dataset class
        # dataset = ECGDataset(self.cfg["dataloader"],
        #                      split="all",
        #                      all_waveforms=True)
        
        # using the data in dataset defined by ourselves
        dataset = self.datasets["all"]
        
        dataloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.cfg["optimizer"]["batch_size"],
                            shuffle=False,
                            num_workers=self.cfg["dataloader"]["n_dataloader_workers"],
                            pin_memory=True,
                            drop_last=False)
        print("running eval on all", flush=True)
        
        # with label
        losses, ys, yhs, score = self.run_epoch("all", dataloader=dataloader)
        print(f"Score of eval on all:{score}")
        
        pd.DataFrame({"y": ys.flatten(), 
                      "yh": yhs.flatten()}).to_csv(
            os.path.join(self.output, "all_waveforms_preds.csv"))
        
        # no label
#         _, _, yhs, _ = self.run_epoch("all", no_label=True, dataloader=dataloader)
        
#         pd.DataFrame({"yh": yhs.flatten()}).to_csv(
#             os.path.join(self.output, "all_waveforms_preds.csv"))

    def try_to_load(self, name="checkpoint.pt", model_path=None):
        try:
            if not model_path:
                model_path = os.path.join(self.output, name)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optim.load_state_dict(checkpoint["opt_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_dict"])
            epoch_resume = checkpoint["epoch"] + 1
            best_score = checkpoint["best_score"]
            train_loss_history = checkpoint["train_loss_history"]
            train_score_histoty = checkpoint["train_score_histoty"]
            # train_sensitivity_history = checkpoint["train_sensitivity_history"]
            # train_specificity_history = checkpoint["train_specificity_history"]
            valid_loss_history = checkpoint["valid_loss_history"]
            valid_score_histoty = checkpoint["valid_score_histoty"]
            # valid_sensitivity_history = checkpoint["valid_sensitivity_history"]
            # valid_specificity_history = checkpoint["valid_specificity_history"]
            print(f"Resuming from epoch {epoch_resume}")
        
        except FileNotFoundError:
            print("Starting run from scratch")
            epoch_resume = 0
            best_score = 0
            train_loss_history, train_score_histoty = [], []
            valid_loss_history, valid_score_histoty = [], []
            
        return epoch_resume, best_score, train_loss_history, train_score_histoty, valid_loss_history, valid_score_histoty

            
            
    def save_model(self, losses, epoch, score, best_score, 
                    train_loss_history, train_score_histoty, 
                   valid_loss_history, valid_score_histoty, save_all=False):
        
        # mean loss of each epoch
        loss = np.mean(losses)
        
        # save best score
        best = False
        if best_score < score:
            best_score = score
            best = True

        # Save checkpoint
        save = {
            "epoch": epoch,
            "loss": loss,
            "best_score": best_score,
            # "sensitivity": sensitivity,
            # "specificity": specificity,
            "train_loss_history": train_loss_history,
            "train_score_histoty": train_score_histoty,
            # "train_sensitivity_history": train_sensitivity_history,
            # "train_specificity_history": train_specificity_history,
            "valid_loss_history": valid_loss_history,
            "valid_score_histoty": valid_score_histoty,
            # "valid_sensitivity_history": valid_sensitivity_history,
            # "valid_specificity_history": valid_specificity_history,
            "state_dict": self.model.state_dict(),
            "opt_dict": self.optim.state_dict(),
            "scheduler_dict": self.scheduler.state_dict(),
        }

        torch.save(save, os.path.join(self.output, "checkpoint.pt"))
        
        if best:
            torch.save(save, os.path.join(self.output, "best.pt"))
            
        if save_all:
            torch.save(save, os.path.join(self.output, "checkpoint{}.pt".format(epoch)))
            
        return best_score


def optimizer_and_scheduler(cfg, model):
    """
    A function to build a torch optimizer and scheduler.
    Arguments:
        cfg: The config dict, like config.py's cfg["optimizer"].
        model: The torch model to train.
    """
    if cfg["optimizer"] == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"])
    else:
        optim = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=.9,
            weight_decay=cfg["weight_decay"])

    if cfg["reduce_on_plateau"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=cfg["patience"])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, cfg["n_epochs"] / cfg["lr_plateaus"])
    return optim, scheduler
