from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class CnnResult:
    model: nn.Module
    accuracy: float
    f1_macro: float
    best_epoch: int


class SequenceNormalizer:
    def __init__(self, mean, std):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, X):
        mean = X.mean(axis=(0, 1), keepdims=True)
        std = X.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean, std)

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


class HAR1DCNN(nn.Module):
    def __init__(self, n_features=6, n_classes=6, n_channels=None, variant="small"):
        super().__init__()
        if n_channels is not None:
            n_features = n_channels
        self.register_buffer("feature_mean", torch.zeros(1, 1, n_features))
        self.register_buffer("feature_std", torch.ones(1, 1, n_features))
        if variant == "small":
            self.net = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            head_in = 128
        elif variant == "improved":
            self.net = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=9, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            head_in = 128
        else:
            raise ValueError("variant must be 'small' or 'improved'")
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(head_in, n_classes),
        )

    def set_normalization(self, mean, std):
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.feature_std.copy_(torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1))

    def forward(self, x):
        x = (x - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        x = x.transpose(1, 2)
        return self.classifier(self.net(x))


def _device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(y, n_classes, device):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_epochs(model, X, y, epochs, batch_size, device, lr=1e-3, patience=None, X_val=None, y_val=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y, 6, device))
    best_state = None
    best_acc = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in _make_loader(X, y, batch_size=batch_size, shuffle=True):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if X_val is None or y_val is None:
            best_epoch = epoch
            continue

        preds = predict_cnn(model, X_val, device=device)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


def train_cnn_candidate(
    X,
    y,
    groups,
    epochs=30,
    batch_size=128,
    patience=5,
    device=None,
    seed=42,
    variant="small",
    normalize=False,
):
    _seed_everything(seed)
    device = _device(device)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    normalizer = None
    X_train_fit = X_train
    X_val_fit = X_val
    if normalize:
        normalizer = SequenceNormalizer.fit(X_train_fit)
        X_train_fit = normalizer.transform(X_train_fit)
        X_val_fit = normalizer.transform(X_val_fit)

    model = HAR1DCNN(n_features=X.shape[2], n_classes=6, variant=variant)
    if not normalize:
        model.set_normalization(X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1)))
    model.sequence_normalizer = normalizer
    best_epoch = _train_epochs(
        model,
        X_train_fit,
        y_train,
        epochs,
        batch_size,
        device,
        patience=patience,
        X_val=X_val_fit,
        y_val=y_val,
    )

    preds = predict_cnn(model, X_val_fit, device=device)
    return CnnResult(
        model=model.cpu(),
        accuracy=accuracy_score(y_val, preds),
        f1_macro=f1_score(y_val, preds, average="macro", zero_division=0),
        best_epoch=best_epoch,
    )


def fit_cnn_full(X, y, epochs=30, batch_size=128, device=None, seed=42, variant="small", normalize=False):
    _seed_everything(seed)
    device = _device(device)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    normalizer = None
    X_fit = X
    if normalize:
        normalizer = SequenceNormalizer.fit(X_fit)
        X_fit = normalizer.transform(X_fit)
    model = HAR1DCNN(n_features=X.shape[2], n_classes=6, variant=variant)
    if not normalize:
        model.set_normalization(X.mean(axis=(0, 1)), X.std(axis=(0, 1)))
    model.sequence_normalizer = normalizer
    _train_epochs(model, X_fit, y, epochs, batch_size, device)
    return model.cpu()


def predict_cnn(model, X, batch_size=512, device=None, normalize=False):
    device = _device(device)
    model = model.to(device)
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    if normalize:
        normalizer = getattr(model, "sequence_normalizer", None)
        if normalizer is None:
            raise ValueError("normalize=True requires a model fitted with normalize=True")
        X = normalizer.transform(X)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)
