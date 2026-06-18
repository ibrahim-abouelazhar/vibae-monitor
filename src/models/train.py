import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

from src.data_loader import VibrationPreprocessor
from src.models.autoencoder import get_model
from src.models.classifier import get_classifier

LABEL_TO_IDX = {
    'Normal': 0,
    'Ball_007': 1, 'Ball_014': 2, 'Ball_021': 3,
    'IR_007': 4, 'IR_014': 5, 'IR_021': 6,
    'OR_007': 7, 'OR_014': 8, 'OR_021': 9
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

class VibrationDataset(Dataset):
    def __init__(self, scaled_magnitudes, augment=False, noise_factor=0.03):
        # Input shape: (n_samples, 128, 32)
        self.data = torch.tensor(scaled_magnitudes, dtype=torch.float32).unsqueeze(1)
        self.augment = augment
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.augment:
            # Data Augmentation: add random noise
            noise = torch.randn_like(x) * self.noise_factor
            x = torch.clamp(x + noise, 0.0, 1.0)
        return x, x


class SpectrogramDataset(Dataset):
    def __init__(self, data, labels, label_map=LABEL_TO_IDX, augment=False, noise_factor=0.03):
        if data.ndim == 3 and data.shape[1:] == (128, 32):  # FIXED P6
            data = data[:, :32, :]  # FIXED P6
        # Input shape: (n_samples, 32, 32)
        # Add channel dimension -> (n_samples, 1, 32, 32)
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
        self.augment = augment
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.augment:
            noise = torch.randn_like(x) * self.noise_factor
            x = torch.clamp(x + noise, 0.0, 1.0)
        return x, y


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(
    normal_mat_path="data/raw/Time_Normal_1_098.mat",
    processed_dir="data/processed",
    model_type="cnn",
    epochs=40,
    batch_size=64,
    learning_rate=0.001,
    window_size=2048,  # FIXED P2
    step_size=512,
    fs=48000,  # FIXED P1
    device=None,
    patience=5,
    augment=True,
    register_model=True
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Autoencoder. Device: {device}")

    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load normal vibration data
    preprocessor = VibrationPreprocessor(window_size=window_size, step_size=step_size, fs=fs)  # FIXED P1 P2
    normal_signal = preprocessor.load_de_signal(normal_mat_path)
    windows = preprocessor.segment_signal(normal_signal)
    
    # 2. Extract STFT Magnitudes
    magnitudes = []
    for w in windows:
        mag, _ = preprocessor.compute_stft(w)
        magnitudes.append(mag)
    magnitudes = np.array(magnitudes)  # Shape: (n_samples, 128, 32)
    
    # 3. Split train/validation (80/20)
    split_idx = int(0.8 * len(magnitudes))
    indices = np.arange(len(magnitudes))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_mags = magnitudes[train_indices]
    val_mags = magnitudes[val_indices]
    
    # 4. Fit scaler
    preprocessor.scaler.fit(train_mags.reshape(len(train_mags), -1))
    preprocessor.is_fitted = True
    
    train_scaled = preprocessor.transform_magnitude(train_mags)
    val_scaled = preprocessor.transform_magnitude(val_mags)
    
    # Save scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    preprocessor.save_scaler(scaler_path)
    
    # Datasets
    train_dataset = VibrationDataset(train_scaled, augment=augment)
    val_dataset = VibrationDataset(val_scaled, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Model setup
    model = get_model(model_type, height=128, width=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    mlflow.set_experiment("VibAE-Monitor-2D")
    
    early_stopping = EarlyStopping(patience=patience)
    best_val_loss = float("inf")
    
    with mlflow.start_run(run_name=f"Autoencoder_{model_type}") as run:
        mlflow.log_param("model_family", "autoencoder")
        mlflow.log_param("model_type", f"2d_{model_type}")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("data_augmentation", augment)
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)
            
            # Val
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_x)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights_path = os.path.join(processed_dir, "best_model.pt")
                torch.save(model.state_dict(), best_weights_path)
                
            # Early Stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model weights for threshold calculations
        model.load_state_dict(torch.load(os.path.join(processed_dir, "best_model.pt")))
        model.eval()
        
        # Calculate thresholds
        val_mses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                sample_mses = torch.mean((batch_x - outputs) ** 2, dim=[1, 2, 3])
                val_mses.extend(sample_mses.cpu().numpy())
                
        val_mses = np.array(val_mses)
        mean_mse = np.mean(val_mses)
        std_mse = np.std(val_mses)
        threshold = mean_mse + 3 * std_mse
        
        mlflow.log_metric("val_mse_mean", mean_mse)
        mlflow.log_metric("val_mse_std", std_mse)
        mlflow.log_metric("anomaly_threshold", threshold)
        
        # Save threshold config
        threshold_config = {
            "model_type": model_type,
            "window_size": window_size,
            "step_size": step_size,
            "fs": fs,
            "height": 128,
            "width": 32,
            "mean_mse": float(mean_mse),
            "std_mse": float(std_mse),
            "threshold": float(threshold)
        }
        
        threshold_path = os.path.join(processed_dir, "threshold.json")
        with open(threshold_path, "w") as f:
            json.dump(threshold_config, f, indent=4)
            
        # Log to model registry
        if register_model:
            mlflow.pytorch.log_model(
                model, 
                artifact_path="autoencoder_2d_model", 
                registered_model_name=f"VibAE-Autoencoder-{model_type}"
            )
            
    return threshold_config


def train_classifier(
    npz_path="data/CWRU_48k_load_1_CNN_data.npz",
    processed_dir="data/processed",
    epochs=30,
    batch_size=64,
    learning_rate=0.001,
    device=None,
    patience=5,
    augment=True,
    register_model=True
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training CNN Classifier. Device: {device}")

    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load data
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ dataset not found: {npz_path}")
        
    npz_data = np.load(npz_path)
    X = npz_data["data"]      # Shape: (4600, 32, 32)
    y = npz_data["labels"]    # Shape: (4600,)
    
    # 2. Min-max scaling per spectrogram
    X_min = X.min(axis=(1, 2), keepdims=True)
    X_max = X.max(axis=(1, 2), keepdims=True)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
    
    # 3. Train/validation split (80/20) stratified
    # Shuffle indices
    indices = np.arange(len(X_scaled))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(indices))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_x, train_y = X_scaled[train_idx], y[train_idx]
    val_x, val_y = X_scaled[val_idx], y[val_idx]
    
    # Datasets
    train_dataset = SpectrogramDataset(train_x, train_y, augment=augment)
    val_dataset = SpectrogramDataset(val_x, val_y, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model setup
    model = get_classifier(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    mlflow.set_experiment("VibAE-Monitor-2D")
    
    early_stopping = EarlyStopping(patience=patience)
    best_val_acc = 0.0
    best_weights_path = os.path.join(processed_dir, "best_classifier.pt")
    
    with mlflow.start_run(run_name="CNN_Classifier") as run:
        mlflow.log_param("model_family", "classifier")
        mlflow.log_param("model_type", "2d_cnn_classifier")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("data_augmentation", augment)
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
            train_loss /= len(train_loader.dataset)
            train_acc = correct / total
            
            # Val
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
                    
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_weights_path)
                
            # Early Stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        # Save config
        classifier_config = {
            "model_type": "cnn_classifier",
            "num_classes": 10,
            "labels_map": LABEL_TO_IDX,
            "best_val_accuracy": float(best_val_acc)
        }
        
        config_path = os.path.join(processed_dir, "classifier_config.json")
        with open(config_path, "w") as f:
            json.dump(classifier_config, f, indent=4)
            
        if register_model:
            mlflow.pytorch.log_model(
                model,
                artifact_path="classifier_2d_model",
                registered_model_name="VibAE-Classifier-CNN"
            )
            
    return classifier_config


def train_densenet_classifier(
    npz_path="data/CWRU_48k_load_1_CNN_data.npz",
    processed_dir="data/processed",
    device=None,
    batch_size=32,
    phase1_epochs=20,
    phase2_epochs=15,
    patience=5,
    augment=True,
    register_model=False,
):
    """
    Fine-tune DenseNet121 on CWRU spectrograms with progressive unfreezing.
    Phase 1 — only classification head trained (backbone frozen).
    Phase 2 — last dense block + head trained (lower learning rate).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DenseNet] Training on device: {device}")

    os.makedirs(processed_dir, exist_ok=True)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ dataset not found: {npz_path}")

    npz_data = np.load(npz_path)
    X = npz_data["data"]   # (4600, 32, 32)
    y = npz_data["labels"] # (4600,)

    X_min = X.min(axis=(1, 2), keepdims=True)
    X_max = X.max(axis=(1, 2), keepdims=True)
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)

    indices = np.arange(len(X_scaled))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_x, train_y = X_scaled[indices[:split]], y[indices[:split]]
    val_x,   val_y   = X_scaled[indices[split:]], y[indices[split:]]

    train_loader = DataLoader(SpectrogramDataset(train_x, train_y, augment=augment),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SpectrogramDataset(val_x, val_y, augment=False),
                              batch_size=batch_size, shuffle=False)

    model = get_classifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    best_weights_path = os.path.join(processed_dir, "best_classifier.pt")
    best_val_acc = 0.0

    def run_phase(phase_name, n_epochs, lr):
        nonlocal best_val_acc
        print(f"\n[DenseNet] {phase_name} — lr={lr} epochs={n_epochs}")
        optimizer = torch.optim.Adam(model.trainable_params(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        stopper = EarlyStopping(patience=patience)

        for epoch in range(n_epochs):
            model.train()
            correct = total = 0
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)
                _, pred = out.max(1)
                correct += pred.eq(by).sum().item()
                total += by.size(0)
            train_loss /= len(train_loader.dataset)
            train_acc = correct / total

            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_loss += criterion(out, by).item() * bx.size(0)
                    _, pred = out.max(1)
                    val_correct += pred.eq(by).sum().item()
                    val_total += by.size(0)
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total

            print(f"  Epoch {epoch+1:>2}/{n_epochs} | "
                  f"Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                  f"Val {val_loss:.4f} Acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_weights_path)
                print(f"  -> saved (val_acc={val_acc:.4f})")

            scheduler.step()
            stopper(val_loss)
            if stopper.early_stop:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # Phase 1 — backbone frozen, head only
    model.freeze_all()
    run_phase("Phase 1 (head only)", phase1_epochs, lr=1e-3)

    # Phase 2 — last block + head
    model.unfreeze_last_block()
    run_phase("Phase 2 (last block + head)", phase2_epochs, lr=1e-4)

    # Save config (marks model as densenet for model_service)
    mlflow.set_experiment("VibAE-Monitor-2D")
    classifier_config = {
        "model_type": "densenet_classifier",
        "num_classes": 10,
        "labels_map": LABEL_TO_IDX,
        "best_val_accuracy": float(best_val_acc),
    }
    config_path = os.path.join(processed_dir, "classifier_config.json")
    with open(config_path, "w") as f:
        json.dump(classifier_config, f, indent=4)

    if register_model:
        model.load_state_dict(torch.load(best_weights_path, map_location="cpu"))
        model.eval()
        with mlflow.start_run(run_name="DenseNet_Classifier"):
            mlflow.log_params({"model_family": "densenet_classifier",
                               "best_val_accuracy": best_val_acc})
            mlflow.pytorch.log_model(model, "densenet_classifier",
                                     registered_model_name="VibAE-DenseNet-Classifier")

    print(f"\n[DenseNet] Training complete — best val acc: {best_val_acc:.4f}")
    return classifier_config


def hyperparameter_search():
    """
    Run a simple grid search over hyperparameters and log all runs in MLflow.
    """
    print("Starting Hyperparameter Search...")
    learning_rates = [0.001, 0.005]
    batch_sizes = [32, 64]
    
    # 1. Search for Autoencoder
    best_ae_loss = float("inf")
    best_ae_lr = None
    best_ae_bs = None
    
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"Grid search (Autoencoder) - LR: {lr}, BS: {bs}")
            # Train model and disable MLflow registration for search runs to avoid spamming the registry
            config = train_model(
                epochs=10, 
                learning_rate=lr, 
                batch_size=bs, 
                register_model=False,
                patience=3
            )
            
            # Read latest metric
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("VibAE-Monitor-2D")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="params.model_family = 'autoencoder'",
                order_by=["metrics.val_loss ASC"],
                max_results=1
            )
            if runs:
                val_loss = runs[0].data.metrics.get("val_loss", float("inf"))
                if val_loss < best_ae_loss:
                    best_ae_loss = val_loss
                    best_ae_lr = lr
                    best_ae_bs = bs
                    
    print(f"Best Autoencoder Config found - LR: {best_ae_lr}, BS: {best_ae_bs} with Val Loss: {best_ae_loss:.6f}")
    
    # Retrain and register the best autoencoder
    train_model(
        epochs=30,
        learning_rate=best_ae_lr,
        batch_size=best_ae_bs,
        register_model=True
    )
    
    # 2. Search for Classifier
    best_clf_acc = 0.0
    best_clf_lr = None
    best_clf_bs = None
    
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"Grid search (Classifier) - LR: {lr}, BS: {bs}")
            config = train_classifier(
                epochs=10,
                learning_rate=lr,
                batch_size=bs,
                register_model=False,
                patience=3
            )
            acc = config.get("best_val_accuracy", 0.0)
            if acc > best_clf_acc:
                best_clf_acc = acc
                best_clf_lr = lr
                best_clf_bs = bs
                
    print(f"Best Classifier Config found - LR: {best_clf_lr}, BS: {best_clf_bs} with Val Acc: {best_clf_acc:.4f}")
    
    # Retrain and register the best classifier
    train_classifier(
        epochs=30,
        learning_rate=best_clf_lr,
        batch_size=best_clf_bs,
        register_model=True
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 2D Spectrogram Autoencoder & Classifier")
    parser.add_argument("--mode", type=str, default="all", choices=["autoencoder", "classifier", "densenet", "search"], help="Execution mode")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    if args.mode == "autoencoder":
        train_model(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
    elif args.mode == "classifier":
        train_classifier(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
    elif args.mode == "densenet":
        train_densenet_classifier()
    elif args.mode == "search":
        hyperparameter_search()
    else: # all
        train_model(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
        train_classifier(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
