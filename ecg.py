"""A function for training and evaluating ECG models."""

import copy
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import scipy.signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
from dataset import ECGDataset
# from dataset_new import ECGDataset
from train import Trainer, optimizer_and_scheduler
from model import ECGModel

def ecg(
        mode="train",
        task="cvm",
        eval_model_path=".",
        eval_model_name="best.pt",
        use_label_file_for_eval=False,
        cfg_updates={},
        log_path="",
        datasets=None,
        cfg =None
    ):
    """
    A function to train and evaluate ECG models.
    Arguments:
        mode: Either "train" or "eval".
        task: The task to train or eval on. Should be a key in config.py's task_cfg.
        eval_model_path: Only used in eval mode. Path to the directory for evaluation.
        eval_model_name: Only used in eval mode. Name of the model to evaluate.
        use_label_file_for_eval: Only used in eval mode. If true, load a label file;
            if false, evaluate on all files in a directory.
        cfg_updates: A nested dict whose schema is a partial copy of config.py's cfg.
            This dict will be used to overwrite the elements in cfg, and to name directories
            during training.
        log_path: Mainly for use with Stanford infrastructure. If set, the file at this path
            will be copied to the model directory after training/evaluation.

    For training, set `task` and overwrite any hyperparameters in `cfg_updates`.

    For evaluation, set `mode="eval"`, set `task` and overwrite any hyperparameters in `cfg_updates`,
    and set `eval_model_path`, `eval_model_name`, and `use_label_file_for_eval` as necessary.
    """

    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    np.random.seed(0)

    # configaration
    # cfg = config.update_config_dict(
    #     config.cfg,
    #     config.task_cfg[task],
    # )
    # cfg = config.update_config_dict(
    #     config.cfg,
    #     cfg_updates,
    # )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # save_path   
    # model_name = f"{task},{config.dict_to_str(cfg_updates)}"
    # print(model_name, flush=True)
    
    if mode == "train":
        output = cfg["optimizer"]["save_path"]
        os.makedirs(output, exist_ok=True)
    else:
        output = eval_model_path 
        
    # model
    model = ECGModel(
        cfg["model"],
        num_input_channels=(len(cfg["dataloader"]["leads"])
            if cfg["dataloader"]["leads"] else 12),
        num_outputs=len(cfg["dataloader"]["label_keys"]),
        binary=cfg["dataloader"]["binary"]
    ).float()

    # put model into device
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    # optimizer
    optim, scheduler = optimizer_and_scheduler(cfg["optimizer"], model)
    
    # dataset
    if datasets==None:
        if mode == "train":

            # using the new dataset defined by myself
            # datasets = {k: ECGDataset(cfg["dataloader"], k) for k in ["train", "valid", "test", "all"]}

            # the original defined dataset
            # datasets = {k: ECGDataset(cfg["dataloader"], k, output=output, all_waveforms=True) for k in ["train", "valid", "test", "all"]}

            # using TensorDataset to creat dataset directly
            waveform_file_path = cfg["dataloader"]["waveforms_file"]
            ecg_waveforms = np.load(waveform_file_path).astype(np.float32)

            labels_file_path = cfg["dataloader"]["label_file"]
            ecg_labels = np.array(pd.read_csv(labels_file_path)[cfg["dataloader"]["label_keys"]])

            # do some preprocess to the waveforms data
            if (cfg["dataloader"]["normalize_y"] or cfg["dataloader"]["notch_filter"] 
                or cfg["dataloader"]["baseline_filter"] or cfg["dataloader"]["mean_zero"]):

                for i in range(len(ecg_waveforms)):

                    # baseline filter
                    if cfg["dataloader"]["baseline_filter"]:
                        ecg_waveforms[i] = baseline(ecg_waveforms[i])

                    # notch filter
                    if cfg["dataloader"]["notch_filter"]:
                        ecg_waveforms[i] = notch(ecg_waveforms[i])

                    # normalization
                    if cfg["dataloader"]["normalize_y"]:
                        ecg_waveforms[i] = normalize_data(ecg_waveforms[i])

                    # zero mean
                    if cfg["dataloader"]["mean_zero"]:
                        for lead in range(len(ecg_waveforms[i])):
                            ecg_waveforms[i][lead] = ecg_waveforms[i][lead] - np.mean(ecg_waveforms[i][lead])

                print('data preprocessed!')

            # Step 1: Split into training and test data 
            temp_data, test_data, temp_labels, test_labels = train_test_split(ecg_waveforms, ecg_labels, test_size = 0.2, random_state=42)

            # Step 2: Split the temporary data into training and validation sets
            train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=0.25, random_state=42)

            waveforms = {"train": train_data,
                        "test": test_data,
                        "valid": val_data,
                        "all": ecg_waveforms}

            labels = {"train": train_labels,
                    "test": test_labels,
                    "valid": val_labels,
                     "all": ecg_labels}

            # creat datasets
            datasets = {k: TensorDataset(torch.tensor(waveforms[k], dtype=torch.float32), 
                                         torch.tensor(labels[k], dtype=torch.float32)) 
                        for k in ["train", "valid", "test", "all"]}
        
    # create dataloaders
    dataloaders = {
        key:
            torch.utils.data.DataLoader(
                datasets[key],
                batch_size=cfg["optimizer"]["batch_size"],
                num_workers=cfg["dataloader"]["n_dataloader_workers"],
                shuffle=(key == "train"),
                drop_last=(key == "train"),
                pin_memory=True,
            )
        for key in ["train", "valid", "test", "all"]}
    
    # trainer
    trainer = Trainer(
                cfg,
                device,
                model,
                optim,
                scheduler,
                datasets,
                dataloaders,
                output,
    )

    # train
    if mode == "train":
        trainer.train()

    elif mode == "eval":
        best_epoch, best_score = trainer.try_to_load("best.pt")
        print(f"Best score seen: {best_score:.3f} at epoch {best_epoch}", flush=True)
        if use_label_file_for_eval:
            trainer.run_eval_on_split("test", report_performance=True)
        else:
            trainer.run_eval_on_all()

    if log_path:
        shutil.copy(log_path, output)


# normalization
def normalize_data(data):
    """
    Normalize a signal using min-max normalization.
    
    Parameters:
    - signal: Input signal (numpy array or list)
    
    Returns:
    - normalized_signal: Normalized signal
    """
    
    row, __ = data.shape
    processed_data = np.zeros(data.shape)
    
    for lead in range(row):
        # Calculate the minimum and maximum values of the signal
        min_val = np.min(data[lead])
        max_val = np.max(data[lead])

        # Perform min-max normalization
        normalized_signal = (data[lead] - min_val) / (max_val - min_val)
        
        processed_data[lead] = normalized_signal
    
    return processed_data

# baseline 
def baseline(data):
    row,__ = data.shape
    sampling_frequency = 500

    win_size = int(np.round(0.2 * sampling_frequency)) + 1
    baseline = scipy.ndimage.median_filter(data, [1, win_size], mode="constant")
    win_size = int(np.round(0.6 * sampling_frequency)) + 1
    baseline = scipy.ndimage.median_filter(baseline, [1, win_size], mode="constant")
    filt_data = data - baseline

    return filt_data

# notch filter
def notch(data):
    sampling_frequency = 500
    row, __ = data.shape
    processed_data = np.zeros(data.shape)
    b = np.ones(int(0.02 * sampling_frequency)) / 50.
    a = [1]
    for lead in range(0, row):
        X = scipy.signal.filtfilt(b, a, data[lead,:])
        processed_data[lead,:] = X
        
    return processed_data
