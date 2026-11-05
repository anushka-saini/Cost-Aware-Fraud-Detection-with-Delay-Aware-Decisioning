"""Neural networks for tabular data, all trained with the same loop.

Four architectures:

* ``mlp``: plain feed-forward network.
* ``resnet``: the ResNet-style MLP from Gorishniy et al., "Revisiting Deep
  Learning Models for Tabular Data" (NeurIPS 2021).
* ``ft_transformer``: the FT-Transformer from the same paper. Every feature
  becomes a small vector ("token") and a transformer looks at all of them.
* ``tabm``: TabM from Gorishniy et al. (ICLR 2025), from the ``tabm`` package.
  One network that behaves like k MLPs trained together; its prediction is
  the average of the k.

Inputs are log-scaled and standardised with ``SignedLogScaler`` (training
statistics only). The loss is a weighted binary cross-entropy: fraud rows are
multiplied by ``pos_weight`` and every row by its optional sample weight.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from fraud.models import SignedLogScaler

# ---------------------------------------------------------------------------
# Architectures. Each one returns logits of shape (batch, k): k = 1 except TabM.
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_inputs, width=128, n_layers=3, dropout=0.1):
        super().__init__()
        layers, size = [], n_inputs
        for _ in range(n_layers):
            layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
            size = width
        layers.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(width), nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResNet(nn.Module):
    def __init__(self, n_inputs, width=128, n_blocks=2, hidden_factor=2, dropout=0.1):
        super().__init__()
        self.first = nn.Linear(n_inputs, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, width * hidden_factor, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.head(self.blocks(self.first(x)))


class FTTransformer(nn.Module):
    def __init__(self, n_inputs, d_token=32, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        # Feature tokenizer: feature j with value x becomes x * weight[j] + bias[j].
        scale = 1 / np.sqrt(d_token)
        self.weight = nn.Parameter(torch.empty(n_inputs, d_token).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.empty(n_inputs, d_token).uniform_(-scale, scale))
        # An extra learned [CLS] token; its output is used for the prediction.
        self.cls = nn.Parameter(torch.empty(1, 1, d_token).uniform_(-scale, scale))
        layer = nn.TransformerEncoderLayer(d_token, n_heads, dim_feedforward=d_token * 2, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias            # (batch, features, d_token)
        cls = self.cls.expand(len(x), -1, -1)
        out = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(out[:, 0])


class TabMNet(nn.Module):
    def __init__(self, n_inputs, width=256, n_blocks=2, dropout=0.1, k=32):
        super().__init__()
        import tabm

        self.net = tabm.TabM.make(n_num_features=n_inputs, d_out=1, k=k, n_blocks=n_blocks,
                                  d_block=width, dropout=dropout)

    def forward(self, x):
        return self.net(x).squeeze(-1)                                 # (batch, k)


ARCHITECTURES = {"mlp": MLP, "resnet": ResNet, "ft_transformer": FTTransformer, "tabm": TabMNet}

TRAINING_DEFAULTS = dict(learning_rate=1e-3, weight_decay=1e-5, batch_size=4096, max_epochs=30, patience=4)


class NeuralNetClassifier:
    """Fit / predict wrapper around one of the architectures above.

    With validation data, training stops when validation PR-AUC has not
    improved for ``patience`` epochs and the best epoch is kept
    (``best_epoch_``). Without validation data it trains for exactly
    ``epochs`` epochs; that is how the final models are refitted.
    """

    def __init__(self, architecture: str, seed: int = 42, pos_weight: float = 20, epochs: int | None = None,
                 learning_rate=None, weight_decay=None, batch_size=None, max_epochs=None, patience=None,
                 **architecture_params):
        self.architecture, self.seed, self.pos_weight, self.epochs = architecture, seed, pos_weight, epochs
        given = dict(learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size,
                     max_epochs=max_epochs, patience=patience)
        self.training = {**TRAINING_DEFAULTS, **{k: v for k, v in given.items() if v is not None}}
        self.architecture_params = architecture_params

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, verbose=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        settings = self.training

        self.scaler_ = SignedLogScaler().fit(X)
        inputs = torch.from_numpy(self.scaler_.transform(X))
        labels = torch.tensor(np.asarray(y), dtype=torch.float32)
        weights = torch.ones(len(labels)) if sample_weight is None else torch.tensor(np.asarray(sample_weight),
                                                                                      dtype=torch.float32)
        weights = weights * torch.where(labels == 1, float(self.pos_weight), 1.0)

        self.net_ = ARCHITECTURES[self.architecture](inputs.shape[1], **self.architecture_params)
        optimiser = torch.optim.AdamW(self.net_.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        generator = torch.Generator().manual_seed(self.seed)

        with_validation = X_val is not None
        n_epochs = settings["max_epochs"] if with_validation else self.epochs
        if n_epochs is None:
            raise ValueError("Without validation data, pass epochs=")

        best_score, best_state, epochs_without_gain = -1.0, None, 0
        self.history_ = []
        for epoch in range(1, n_epochs + 1):
            self._train_one_epoch(inputs, labels, weights, optimiser, generator, settings["batch_size"])
            if not with_validation:
                continue
            score = float(average_precision_score(np.asarray(y_val), self.predict_proba(X_val)[:, 1]))
            self.history_.append(score)
            if verbose:
                print(f"    epoch {epoch:2d}  validation PR-AUC = {score:.4f}", flush=True)
            if score > best_score:
                best_score, self.best_epoch_, epochs_without_gain = score, epoch, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= settings["patience"]:
                    break

        if with_validation:
            self.net_.load_state_dict(best_state)
            self.validation_pr_auc_ = best_score
        else:
            self.best_epoch_ = n_epochs
        return self

    def _train_one_epoch(self, inputs, labels, weights, optimiser, generator, batch_size):
        self.net_.train()
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if len(batch) < 2:            # batch norm needs at least two rows
                continue
            logits = self.net_(inputs[batch])                          # (batch, k)
            target = labels[batch].unsqueeze(1).expand_as(logits)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=1)
            loss = (loss * weights[batch]).sum() / weights[batch].sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    def predict_proba(self, X, batch_size=65536):
        inputs = torch.from_numpy(self.scaler_.transform(X[self.scaler_.columns_]))
        self.net_.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                logits = self.net_(inputs[start:start + batch_size])
                parts.append(torch.sigmoid(logits).mean(dim=1))       # TabM: average of its k members
        p = torch.cat(parts).numpy().astype(np.float64)
        return np.column_stack([1 - p, p])
