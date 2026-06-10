"""
trainer.py — Real PyTorch Training Engine
Builds models from parsed config, trains on user data, streams metrics.
"""

import io
import json
import math
from typing import Generator

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from parser import ParsedModel
from diagnostics import Suggestion


# ─────────────────────────────────────────────
#  DATASET PREPARATION
# ─────────────────────────────────────────────

def prepare_dataset(data_text: str, task_type: str = "classification"):
    """
    Parse CSV/JSON text into PyTorch DataLoaders.
    Auto-detects the target column (last column).
    Returns: (train_loader, val_loader, input_dim, output_dim, task_type)
    """
    # Try CSV first, then JSON
    try:
        df = pd.read_csv(io.StringIO(data_text))
    except Exception:
        try:
            df = pd.read_json(io.StringIO(data_text))
        except Exception:
            # Try to handle raw comma-separated lines
            lines = data_text.strip().split("\n")
            if len(lines) < 2:
                raise ValueError("Need at least 2 rows of data")
            # Check if first line is header
            first_line = lines[0].split(",")
            try:
                [float(x) for x in first_line]
                # All numeric — no header
                df = pd.read_csv(io.StringIO(data_text), header=None)
            except ValueError:
                df = pd.read_csv(io.StringIO(data_text))

    if df.shape[0] < 4:
        raise ValueError(f"Need at least 4 rows of data, got {df.shape[0]}")
    if df.shape[1] < 2:
        raise ValueError(f"Need at least 2 columns (features + target), got {df.shape[1]}")

    # Last column is target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Determine task type
    unique_targets = len(set(y))
    if task_type == "auto":
        if unique_targets <= 20 and unique_targets < len(y) * 0.1:
            task_type = "classification"
        else:
            task_type = "regression"

    # Encode target for classification
    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
        output_dim = len(le.classes_)
        if output_dim == 2:
            output_dim = 1  # Binary classification uses sigmoid
    else:
        output_dim = 1
        y = y.astype(float)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.astype(float))

    # Split
    test_size = min(0.2, max(0.1, 2.0 / len(X)))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    if task_type == "classification" and output_dim > 1:
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    batch_size = min(32, len(X_train))
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X.shape[1]
    return train_loader, val_loader, input_dim, output_dim, task_type


# ─────────────────────────────────────────────
#  MODEL BUILDING
# ─────────────────────────────────────────────

def _get_activation(name: str) -> nn.Module:
    """Convert activation name to PyTorch module."""
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
    }
    return activations.get(name, nn.ReLU())


def build_model(
    input_dim: int,
    output_dim: int,
    layer_sizes: list = None,
    activations: list = None,
    use_dropout: bool = False,
    use_batchnorm: bool = False,
    task_type: str = "classification",
) -> nn.Module:
    """
    Build a feedforward neural network from configuration.
    """
    if not layer_sizes or len(layer_sizes) < 1:
        # Default architecture based on input_dim
        if input_dim <= 10:
            layer_sizes = [32, 16]
        elif input_dim <= 100:
            layer_sizes = [128, 64, 32]
        else:
            layer_sizes = [256, 128, 64]

    activation_name = "relu"
    if activations:
        # Use the first non-'none' activation
        for a in activations:
            if a != "none":
                activation_name = a
                break

    layers = []
    prev_dim = input_dim

    for i, hidden_dim in enumerate(layer_sizes):
        layers.append(nn.Linear(prev_dim, hidden_dim))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(_get_activation(activation_name))

        if use_dropout:
            layers.append(nn.Dropout(0.3))

        prev_dim = hidden_dim

    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))

    # Add sigmoid for binary classification
    if task_type == "classification" and output_dim == 1:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def create_optimizer(model: nn.Module, opt_type: str, lr: float, weight_decay: float = 0.0, momentum: float = 0.0):
    """Create a PyTorch optimizer from config."""
    opt_type_lower = opt_type.lower()
    params = model.parameters()

    if opt_type_lower == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_type_lower == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type_lower == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type_lower == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif opt_type_lower == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    optimizer_inst,
    train_loader: DataLoader,
    criterion: nn.Module,
    task_type: str,
    output_dim: int,
    clip_grad: bool = False,
):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer_inst.zero_grad()
        output = model(X_batch)

        if task_type == "classification" and output_dim == 1:
            output = output.squeeze()
            loss = criterion(output, y_batch)
            preds = (output >= 0.5).float()
            correct += (preds == y_batch).sum().item()
        elif task_type == "classification":
            loss = criterion(output, y_batch)
            preds = output.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
        else:
            output = output.squeeze()
            loss = criterion(output, y_batch)

        # Check for NaN
        if math.isnan(loss.item()):
            return float("nan"), 0.0

        loss.backward()

        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer_inst.step()
        total_loss += loss.item() * len(y_batch)
        total += len(y_batch)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 and task_type == "classification" else 0.0
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    task_type: str,
    output_dim: int,
):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)

            if task_type == "classification" and output_dim == 1:
                output = output.squeeze()
                loss = criterion(output, y_batch)
                preds = (output >= 0.5).float()
                correct += (preds == y_batch).sum().item()
            elif task_type == "classification":
                loss = criterion(output, y_batch)
                preds = output.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
            else:
                output = output.squeeze()
                loss = criterion(output, y_batch)

            if not math.isnan(loss.item()):
                total_loss += loss.item() * len(y_batch)
            total += len(y_batch)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 and task_type == "classification" else 0.0
    return avg_loss, accuracy


def run_comparison(
    parsed: ParsedModel,
    suggestion: Suggestion,
    data_text: str,
    epochs: int = 30,
) -> Generator[dict, None, None]:
    """
    Train two models side by side and yield metrics per epoch.
    Model A: user's current config
    Model B: suggested config
    """
    # Prepare dataset
    train_loader, val_loader, input_dim, output_dim, task_type = prepare_dataset(
        data_text, parsed.task_type
    )

    # Infer layer sizes from parsed model (or use defaults)
    layer_sizes = parsed.layer_sizes if len(parsed.layer_sizes) >= 2 else None

    # Build Model A (current config)
    current_activations = parsed.activations if parsed.activations != ["none"] else ["relu"]
    model_a = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        layer_sizes=layer_sizes,
        activations=current_activations,
        use_dropout=parsed.has_dropout,
        use_batchnorm=parsed.has_batchnorm,
        task_type=task_type,
    )

    # Build Model B (suggested config)
    suggested_acts = suggestion.suggested_activations or ["relu"]
    use_dropout_b = parsed.has_dropout or suggestion.suggested_weight_decay > 0
    use_bn_b = parsed.has_batchnorm or "sigmoid_deep" in [i for i in []]  # use BN if sigmoid issue was found
    model_b = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        layer_sizes=layer_sizes,
        activations=suggested_acts,
        use_dropout=use_dropout_b,
        use_batchnorm=True,  # Suggested model always uses batchnorm
        task_type=task_type,
    )

    # Create optimizers
    current_lr = parsed.lr if parsed.lr else 0.01
    current_momentum = parsed.momentum if parsed.momentum else 0.0
    current_wd = parsed.weight_decay if parsed.weight_decay else 0.0

    opt_a = create_optimizer(
        model_a, parsed.optimizer, current_lr,
        weight_decay=current_wd, momentum=current_momentum
    )
    opt_b = create_optimizer(
        model_b, suggestion.suggested_optimizer, suggestion.suggested_lr,
        weight_decay=suggestion.suggested_weight_decay
    )

    # Loss function
    if task_type == "classification" and output_dim == 1:
        criterion = nn.BCELoss()
    elif task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Determine if grad clipping should be applied
    clip_a = parsed.has_grad_clip
    clip_b = True  # Suggested model always uses clipping

    # Yield initial state
    yield {
        "epoch": 0,
        "total_epochs": epochs,
        "model_a": {
            "name": f"{parsed.optimizer} lr={parsed.lr or 0.01} ({parsed.loss_function})",
            "train_loss": 0, "train_acc": 0,
            "val_loss": 0, "val_acc": 0,
        },
        "model_b": {
            "name": suggestion.good_label,
            "train_loss": 0, "train_acc": 0,
            "val_loss": 0, "val_acc": 0,
        },
        "status": "started",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "task_type": task_type,
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        # Train Model A
        train_loss_a, train_acc_a = train_one_epoch(
            model_a, opt_a, train_loader, criterion, task_type, output_dim, clip_a
        )
        val_loss_a, val_acc_a = evaluate(model_a, val_loader, criterion, task_type, output_dim)

        # Train Model B
        train_loss_b, train_acc_b = train_one_epoch(
            model_b, opt_b, train_loader, criterion, task_type, output_dim, clip_b
        )
        val_loss_b, val_acc_b = evaluate(model_b, val_loader, criterion, task_type, output_dim)

        status = "training"
        if epoch == epochs:
            status = "completed"

        yield {
            "epoch": epoch,
            "total_epochs": epochs,
            "model_a": {
                "name": f"{parsed.optimizer} lr={parsed.lr or 0.01} ({parsed.loss_function})",
                "train_loss": round(train_loss_a, 6) if not math.isnan(train_loss_a) else "NaN",
                "train_acc": round(train_acc_a, 4),
                "val_loss": round(val_loss_a, 6) if not math.isnan(val_loss_a) else "NaN",
                "val_acc": round(val_acc_a, 4),
            },
            "model_b": {
                "name": suggestion.good_label,
                "train_loss": round(train_loss_b, 6),
                "train_acc": round(train_acc_b, 4),
                "val_loss": round(val_loss_b, 6),
                "val_acc": round(val_acc_b, 4),
            },
            "status": status,
        }
