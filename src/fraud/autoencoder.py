"""Label-free anomaly detector: a tabular autoencoder.

Trained on every transaction in the training window without looking at the
fraud label (fraud is about 0.07% of rows, so it barely affects what the
network learns to reconstruct). The anomaly score of a transaction is its
reconstruction error: rows that do not look like ordinary traffic are
reconstructed badly. Because no labels are involved, the detector can be
refreshed on recent traffic while fraud labels are still arriving.
"""

import numpy as np
import torch
import torch.nn as nn

from fraud.models import SignedLogScaler

AE_PARAMS = dict(hidden=(12, 6), bottleneck=3, learning_rate=1e-3, weight_decay=1e-6,
                 batch_size=4096, max_epochs=40, patience=4, max_train_rows=None)


def _autoencoder_net(n_inputs, hidden, bottleneck):
    sizes = [n_inputs, *hidden, bottleneck]
    encoder, decoder = [], []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        encoder += [nn.Linear(a, b), nn.ReLU()]
    reverse = sizes[::-1]
    for i, (a, b) in enumerate(zip(reverse[:-1], reverse[1:], strict=True)):
        decoder.append(nn.Linear(a, b))
        if i < len(reverse) - 2:
            decoder.append(nn.ReLU())
    return nn.Sequential(*encoder[:-1], *decoder)   # no ReLU on the bottleneck itself


class TabularAutoencoder:
    """``fit(X, X_val)`` takes no labels. ``anomaly_score(X)`` is the per-row
    mean squared reconstruction error in the scaled feature space."""

    def __init__(self, seed: int = 42, **params):
        self.seed = seed
        self.params = {**AE_PARAMS, **params}

    def fit(self, X, X_val, verbose: bool = False):
        p = self.params
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.scaler_ = SignedLogScaler().fit(X)
        train = self.scaler_.transform(X)
        self.n_train_rows_available_ = len(train)
        if p["max_train_rows"] and len(train) > p["max_train_rows"]:
            # Uniform subsample, drawn without reference to the label.
            train = train[rng.choice(len(train), p["max_train_rows"], replace=False)]
        self.n_train_rows_used_ = len(train)
        T, T_val = torch.from_numpy(train), torch.from_numpy(self.scaler_.transform(X_val))

        self.net_ = _autoencoder_net(T.shape[1], p["hidden"], p["bottleneck"])
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, stale = np.inf, None, 0
        self.history_ = []
        for epoch in range(1, p["max_epochs"] + 1):
            self.net_.train()
            order = torch.randperm(len(T), generator=generator)
            for i in range(0, len(order), p["batch_size"]):
                batch = T[order[i:i + p["batch_size"]]]
                optimiser.zero_grad()
                loss_fn(self.net_(batch), batch).backward()
                optimiser.step()
            val_loss = float(self._errors(T_val).mean())
            self.history_.append(val_loss)
            if verbose:
                print(f"    epoch {epoch:2d}  validation reconstruction loss = {val_loss:.5f}", flush=True)
            if val_loss < best_loss - 1e-6:
                best_loss, stale, self.best_epoch_ = val_loss, 0, epoch
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if stale >= p["patience"]:
                    break
        self.net_.load_state_dict(best_state)
        self.validation_loss_ = best_loss
        return self

    def _errors(self, T, batch=262144) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            return torch.cat([((self.net_(T[i:i + batch]) - T[i:i + batch]) ** 2).mean(dim=1)
                              for i in range(0, len(T), batch)]).numpy().astype(np.float64)

    def anomaly_score(self, X) -> np.ndarray:
        return self._errors(torch.from_numpy(self.scaler_.transform(X[self.scaler_.columns_])))

    def feature_errors(self, X) -> np.ndarray:
        """Squared reconstruction error per feature (rows x features), for explanations."""
        T = torch.from_numpy(self.scaler_.transform(X[self.scaler_.columns_]))
        self.net_.eval()
        with torch.no_grad():
            return ((self.net_(T) - T) ** 2).numpy().astype(np.float64)

    @property
    def error_columns_(self) -> list[str]:
        return [*self.scaler_.continuous_, *self.scaler_.binary_]
