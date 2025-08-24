
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .config.config import ModelConfig, TrainingConfig, DataConfig, ExperimentConfig
from .utils.helpers import set_seed, get_device
from .data.processor import DataProcessor
from .data.dataset import StockSequenceDataset
from .models.cnn import StockCNN
from .training.trainer import Trainer

class Pipeline:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainingConfig, data_cfg: DataConfig, exp_cfg: ExperimentConfig):
        set_seed(exp_cfg.seed)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.exp_cfg = exp_cfg
        self.device = get_device(exp_cfg.device)

    def run(self):
        # 1) Load + FE + Label
        proc = DataProcessor(self.data_cfg.csv_path, self.data_cfg.window_size, self.data_cfg.stride,
                             self.data_cfg.prediction_horizon, self.data_cfg.binary_threshold)
        df = proc.load_and_prepare()
        # 2) Split
        train_df, val_df, test_df = proc.split(df, self.data_cfg.train_ratio, self.data_cfg.val_ratio)
        feature_cols = [c for c in df.columns if c not in ("Date","target")]
        X_train, y_train = train_df[feature_cols].values, train_df["target"].values
        X_val, y_val = val_df[feature_cols].values, val_df["target"].values

        # 3) Datasets/Dataloaders
        ds_train = StockSequenceDataset(X_train, y_train, self.data_cfg.window_size, self.data_cfg.stride)
        ds_val = StockSequenceDataset(X_val, y_val, self.data_cfg.window_size, stride=1)
        train_loader = DataLoader(ds_train, batch_size=self.train_cfg.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(ds_val, batch_size=self.train_cfg.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

        # 4) Model
        model = StockCNN(
            num_classes=2, base_channels=self.model_cfg.base_channels,
            channel_multiplier=self.model_cfg.channel_multiplier, num_blocks=self.model_cfg.num_blocks,
            dropout=self.model_cfg.dropout_rate, norm=self.model_cfg.norm_type,
            activation=self.model_cfg.activation, attention=self.model_cfg.attention_type
        )

        # 5) Train
        trainer = Trainer(model, self.device, self.train_cfg)
        best = trainer.fit(train_loader, val_loader)
        return best
