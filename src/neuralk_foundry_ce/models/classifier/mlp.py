import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import ClassifierModel
from ...utils.splitting import with_masked_split, Split
from ...config import global_config


class MLPClassifier(ClassifierModel):
    """
    Train a neural network (MLP) classifier with categorical embeddings on tabular data.

    Inputs
    ------
    - X : Feature matrix (pandas DataFrame). Categorical columns must be of dtype 'category'.
    - y : Target labels (for training only).
    - splits : Optional train/val/test split masks.

    Outputs
    -------
    - y_pred : Predicted class labels.
    - y_score : Class probabilities.

    Notes
    -----
    Requires `torch` to be installed. Categorical variables are embedded.
    Supports basic hyperparameter search for activation, dropout, batchnorm, optimizer, etc.
    """
    name = "mlp-classifier"

    def __init__(self):
        super().__init__()

    def init_model(self, config):
        config = copy.copy(config)

        self.categorical_features = config.get('categorical_features', [])

        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
        }
        optimizer_map = {
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }
        config['activation'] = activation_map[config['activation']]
        config['optimizer'] = optimizer_map[config['optimizer']]

        class MLPNet(nn.Module):
            def __init__(self, input_dim, cat_dims, emb_dims, config):
                super().__init__()
                self.embeddings = nn.ModuleList([
                    nn.Embedding(cat_dim, emb_dim)
                    for cat_dim, emb_dim in zip(cat_dims, emb_dims)
                ])

                emb_total = sum(emb_dims)
                input_total = input_dim + emb_total

                hidden_sizes = [2 ** (int(input_total).bit_length() - 1)]
                for _ in range(config['n_hidden_layers'] - 1):
                    hidden_sizes.append(max(hidden_sizes[-1] // 2, 2))

                layers = []
                prev = input_total
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev, h))
                    if config['batchnorm']:
                        layers.append(nn.BatchNorm1d(h))
                    layers.append(config['activation']())
                    layers.append(nn.Dropout(config['dropout']))
                    prev = h

                layers.append(nn.Linear(prev, config['output_dim']))
                self.model = nn.Sequential(*layers)

            def forward(self, x_num, x_cat):
                embs = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
                x = torch.cat([x_num] + embs, dim=1)
                return self.model(x)

        self.config = config
        self.model_class = MLPNet


    def _preprocess(self, X, is_train=False):
        X_cat_vals = []

        for col in self.categorical_features:
            mapping = self.category_mappings_[col]
            include_unknown = self.include_unknown_index[col]
            mapped = X[col].map(mapping)

            if include_unknown:
                mapped = mapped.fillna(0).astype("int64")
            else:
                # Assign -1 to unknowns for debugging / crash if they appear
                if mapped.isna().any():
                    raise ValueError(f"Unknown category in column '{col}' but unknowns are not allowed.")
                mapped = mapped.astype("int64")

            X_cat_vals.append(mapped)
        if X_cat_vals:
            X_cat = np.stack(X_cat_vals, axis=1)
        else:
            X_cat = np.empty((len(X), 0), dtype="int64")
        X_num = X.drop(columns=self.categorical_features).values.astype("float32")
        return torch.tensor(X_num), torch.tensor(X_cat)


    def train(self, X, y, split_mask, splits):
        device = global_config.device

        mask_train = np.isin(split_mask, Split.TRAIN)
        mask_val = np.isin(split_mask, Split.VAL)
        X_train, y_train = X[mask_train], y[mask_train]
        X_val, y_val = X[mask_val], y[mask_val]

        # Should we expect unknown categories ?
        # Here, we wheck directly against the test if there are any. The point is that in
        # real life, most of the time one knows if a feature may contain unseen categories
        # at inference time.

        self.include_unknown_index = {}
        self.category_mappings_ = {}

        for col in self.categorical_features:
            train_cats = set(X_train[col].unique())
            all_cats = set(X[col].unique())
            unseen_cats = all_cats - train_cats

            needs_unknown = bool(unseen_cats)
            self.include_unknown_index[col] = needs_unknown

            categories = list(train_cats)
            if needs_unknown:
                self.category_mappings_[col] = {cat: i + 1 for i, cat in enumerate(categories)}  # 0 = unknown
            else:
                self.category_mappings_[col] = {cat: i for i, cat in enumerate(categories)}

        (X_num_train, X_cat_train), y_train = self._preprocess(X_train, is_train=True), torch.tensor(y_train, dtype=torch.long)
        (X_num_val, X_cat_val), y_val = self._preprocess(X_val, is_train=False), torch.tensor(y_val, dtype=torch.long)

        cat_dims = [
            len(self.category_mappings_[col]) + (1 if self.include_unknown_index[col] else 0)
            for col in self.categorical_features
        ]
        emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

        # Simulate unknowns in columns where they are present
        rate = self.config.get("simulate_unknowns", 0.01)
        for i, col in enumerate(self.categorical_features):
            if self.include_unknown_index[col] and rate > 0:
                mask = torch.rand(X_cat_train.shape[0]) < rate
                X_cat_train[mask, i] = 0  # Unknown token index

        train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = self.model_class(X_num_train.shape[1], cat_dims, emb_dims, self.config).to(device)
        optimizer = self.config['optimizer'](model.parameters(), lr=self.config['lr'])
        criterion = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')
        patience, patience_counter = 5, 0
        for epoch in range(self.config['epochs']):
            model.train()
            for xb_num, xb_cat, yb in train_loader:
                xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb_num, xb_cat)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(X_num_val.to(device), X_cat_val.to(device))
                val_loss = criterion(out_val, y_val.to(device)).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self.model = model.eval()
        del optimizer

    @with_masked_split
    def forward(self, X):
        device = global_config.device
        X_num, X_cat = self._preprocess(X, is_train=False)

        with torch.no_grad():
            logits = self.model(X_num.to(device), X_cat.to(device))
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

        self.extras['y_score'] = probs
        return probs.argmax(axis=1)

    def get_model_params(self, trial, inputs):
        return {
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'silu']),
            'dropout': trial.suggest_categorical('dropout', [0.0, 0.2, 0.5]),
            'batchnorm': trial.suggest_categorical('batchnorm', [True, False]),
            'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 3),
            'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'sgd']),
            'lr': trial.suggest_categorical('lr', [1e-3, 3e-4, 1e-4]),
        }

    def get_fixed_params(self, inputs):
        params = {
            'output_dim': np.unique(inputs['y']).shape[0],
            'epochs': 100,
        }
        if 'categorical_features' in inputs:
            params['categorical_features'] = inputs['categorical_features']
        return params

    def clean_model(self):
        import torch

        del self.model
        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

