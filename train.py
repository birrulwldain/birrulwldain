import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import json
import os
import logging
from datetime import datetime
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/bwalidain/birrulwldain/logs/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('InformerTraining')

# Optimasi IPEX
try:
    import intel_extension_for_pytorch as ipex
    ipex_version = ipex.__version__
    logger.info(f"IPEX dimuat, versi: {ipex_version}")
except ImportError:
    ipex = None
    logger.warning("IPEX tidak tersedia, lanjutkan tanpa optimasi IPEX")

# Definisi elemen target
BASE_ELEMENTS = ["Si", "Al", "Fe", "Ca", "O", "Na", "N", "Ni", "Cr", "Cl"]
REQUIRED_ELEMENTS = [f"{elem}_{ion}" for elem in BASE_ELEMENTS for ion in [1, 2]]

# Konfigurasi parameter
CONFIG = {
    "data": {
        "dataset_path": "/home/bwalidain/birrulwldain/output/spectral_dataset.h5",
        "element_map_path": "/home/bwalidain/birrulwldain/data/element_map.json",
        "train_split": "train",
        "val_split": "validation",
        "max_train_samples": 10000,
        "max_val_samples": 1500,
        "augment_noise_std": 0.01,
    },
    "model": {
        "input_dim": 1,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 3,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "seq_length": 4096,
        "attn_factor": 7,
    },
    "training": {
        "batch_size": 8,  # Ditingkatkan untuk CPU
        "num_epochs": 5,
        "learning_rate": 1e-3,
        "device": "cpu",  # Paksa CPU
        "model_save_path": "/home/bwalidain/birrulwldain/output/informer_model.pth",
        "class_weight_path": "/home/bwalidain/birrulwldain/output/class_weights.pth",
        "class_weight_factor": 2.0,
        "results_dir": "/home/bwalidain/birrulwldain/output"
    }
}

class HDF5SpectrumDataset:
    def __init__(self, file_path, split, element_map, required_elements, max_samples=None, augment_noise_std=0.0):
        self.file_path = file_path
        self.split = split
        self.required_elements = required_elements
        self.max_samples = max_samples
        self.augment_noise_std = augment_noise_std
        self.samples = []
        self.element_map = {}
        for idx, ion in enumerate(required_elements):
            if ion in element_map:
                self.element_map[ion] = idx + 1
        self.num_classes = len(self.element_map) + 1
        self._load_samples()
        self._validate_labels()

    def _load_samples(self):
        try:
            with h5py.File(self.file_path, 'r') as h5_file:
                if self.split not in h5_file:
                    raise KeyError(f"Split '{self.split}' tidak ditemukan.")
                group = h5_file[self.split]
                spectra = group['spectra'][:]
                labels = group['labels'][:]
                for idx in range(len(spectra)):
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
                    if labels[idx].shape[0] == CONFIG["model"]["seq_length"]:
                        self.samples.append((spectra[idx], labels[idx]))
                    else:
                        logger.warning(f"Sample {idx} in {self.split} skipped due to incorrect sequence length ({labels[idx].shape[0]} != {CONFIG['model']['seq_length']})")
        except Exception as e:
            logger.error(f"Gagal memuat dataset {self.split}: {str(e)}")
            raise

    def _validate_labels(self):
        valid_labels = set(self.element_map.values()).union({0})
        unique_labels = set()
        for _, label in self.samples:
            unique_labels.update(np.unique(label))
        invalid_labels = unique_labels - valid_labels
        if invalid_labels:
            logger.warning(f"Invalid labels found in {self.split}: {invalid_labels}. These will be treated as background.")
        missing_labels = valid_labels - unique_labels
        if missing_labels:
            logger.info(f"Labels not present in {self.split}: {missing_labels}")

    def get_data(self):
        inputs = []
        labels = []
        warning_count = 0
        for spectrum, label in self.samples:
            noise = np.random.normal(0, self.augment_noise_std, spectrum.shape) if self.augment_noise_std > 0 else 0
            input_data = (spectrum + noise)[:, np.newaxis].astype(np.float32)
            label_one_hot = np.zeros((CONFIG["model"]["seq_length"], self.num_classes), dtype=np.float32)
            for t in range(CONFIG["model"]["seq_length"]):
                if label[t] == 0:
                    label_one_hot[t, 0] = 1
                elif label[t] in self.element_map.values():
                    label_one_hot[t, label[t]] = 1
                else:
                    warning_count += 1
                    logger.warning(f"Label {label[t]} at time {t} in {self.split} not in element_map, treating as background")
                    label_one_hot[t, 0] = 1
            inputs.append(input_data)
            labels.append(label_one_hot)
        if warning_count > 0:
            logger.warning(f"Total warnings for invalid labels in {self.split}: {warning_count}")
        return np.array(inputs), np.array(labels)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, factor=7):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        d_k = d_model // self.nhead
        Q = self.q_linear(x).view(batch_size, seq_len, self.nhead, d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.nhead, d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.nhead, d_k).transpose(1, 2)
        u = int(self.factor * np.log(seq_len))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        max_scores, _ = torch.max(scores, dim=-1, keepdim=True)
        sparsity_scores = scores.mean(dim=-1) - max_scores.squeeze(-1) / np.log(seq_len)
        _, top_indices = torch.topk(sparsity_scores, u, dim=-1)
        mask_sparse = torch.zeros_like(scores).scatter_(-1, top_indices.unsqueeze(-1).expand(-1, -1, -1, seq_len), 1.0)
        scores = scores * mask_sparse
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_linear(output)


class InformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_length, num_classes, attn_factor):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                ProbSparseSelfAttention(d_model, nhead, dropout, attn_factor),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                )
            ]) for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = src + self.pos_encoder[:, :src.size(1), :]
        for norm1, attn, norm2, ffn in self.encoder_layers:
            src = norm1(src)
            src = attn(src, src_mask) + src
            src = norm2(src)
            src = ffn(src) + src
        output = self.decoder(src)
        return output


# ... [bagian lain tetap sama]
def compute_class_weights(labels, num_classes, weight_factor=1.0):
    flat_labels = labels.reshape(-1)
    class_counts = np.bincount(flat_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = np.power(class_weights, weight_factor)
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32)

def summarize_model(model, config, results_dir):
    summary = ["Model Summary:"]
    total_params = 0

    def count_params(module, name):
        if isinstance(module, torch.nn.Parameter):
            params = module.numel() if module.requires_grad else 0
        else:
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return f"Layer: {name} - {params} parameters"

    summary.append(count_params(model.embedding, f"embedding (Linear({config['model']['input_dim']}, {config['model']['d_model']}))"))
    total_params += sum(p.numel() for p in model.embedding.parameters() if p.requires_grad)

    summary.append(count_params(model.pos_encoder, f"pos_encoder (Parameter(1, {config['model']['seq_length']}, {config['model']['d_model']}))"))
    total_params += model.pos_encoder.numel() if model.pos_encoder.requires_grad else 0

    for i, (norm1, attn, norm2, ffn) in enumerate(model.encoder_layers, 1):
        norm1_params = sum(p.numel() for p in norm1.parameters() if p.requires_grad)
        summary.append(count_params(norm1, f"encoder_layer_{i}.norm1 (LayerNorm({config['model']['d_model']}))"))
        total_params += norm1_params

        attn_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)
        summary.append(count_params(attn, f"encoder_layer_{i}.attn (ProbSparseSelfAttention(d_model={config['model']['d_model']}, nhead={config['model']['nhead']}))"))
        total_params += attn_params

        norm2_params = sum(p.numel() for p in norm2.parameters() if p.requires_grad)
        summary.append(count_params(norm2, f"encoder_layer_{i}.norm2 (LayerNorm({config['model']['d_model']}))"))
        total_params += norm2_params

        ffn_params = sum(p.numel() for p in ffn.parameters() if p.requires_grad)
        summary.append(count_params(ffn, f"encoder_layer_{i}.ffn (Linear({config['model']['d_model']},{config['model']['dim_feedforward']})-ReLU-Dropout({config['model']['dropout']})-Linear({config['model']['dim_feedforward']},{config['model']['d_model']}))"))
        total_params += ffn_params

    summary.append(count_params(model.decoder, f"decoder (Linear({config['model']['d_model']}, {model.decoder.out_features}))"))
    total_params += sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    summary.append(f"Total Parameters: {total_params:,}")
    logger.info("\n".join(summary))

    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    logger.info(f"Model summary saved to {summary_path}")

    return total_params

def plot_lr_schedule(lr_history, num_epochs, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), lr_history, marker='o', linestyle='-', color='b')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.xticks(range(1, num_epochs + 1, 2))
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "learning_rate_schedule.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Learning rate schedule plot saved to {plot_path}")

def plot_training_metrics(val_acc_history, non_bg_acc_history, val_f1_history, train_loss_history, val_loss_history, num_epochs, results_dir):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(range(1, num_epochs + 1), val_acc_history, marker='o', linestyle='-', color='b', label='Validation Accuracy')
    ax1.plot(range(1, num_epochs + 1), non_bg_acc_history, marker='s', linestyle='--', color='g', label='Non-Background Accuracy')
    ax1.plot(range(1, num_epochs + 1), val_f1_history, marker='^', linestyle='-.', color='r', label='Validation F1 Score')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy / F1 Score", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 1)
    ax1.grid(True, which="both", ls="--")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(range(1, num_epochs + 1), train_loss_history, marker='x', linestyle='-', color='m', label='Train Loss')
    ax2.plot(range(1, num_epochs + 1), val_loss_history, marker='d', linestyle='--', color='c', label='Validation Loss')
    ax2.set_ylabel("Loss", color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax2.legend(loc='upper right')

    plt.title("Training Metrics")
    plt.xticks(range(1, num_epochs + 1, 2))
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "training_metrics.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Training metrics plot saved to {plot_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["training"]["learning_rate"],
        total_steps=num_epochs * len(train_loader),
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=100.0,
        final_div_factor=1000.0
    )
    lr_history = []
    val_acc_history = []
    non_bg_acc_history = []
    val_f1_history = []
    train_loss_history = []
    val_loss_history = []
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        try:
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            for batch_idx, (data, target) in enumerate(train_loader):
                logger.debug(f"Processing batch {batch_idx+1}/{len(train_loader)}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                output = output.transpose(1, 2)
                target = torch.argmax(target, dim=-1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
        except Exception as e:
            logger.error(f"Error during training epoch {epoch+1}: {str(e)}")
            raise

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_targets = []
        non_bg_correct = 0
        non_bg_samples = 0
        try:
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    output = output.transpose(1, 2)
                    target = torch.argmax(target, dim=-1)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()
                    pred = torch.argmax(output, dim=1)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                    non_bg_mask = target != 0
                    non_bg_correct += (pred[non_bg_mask] == target[non_bg_mask]).sum().item()
                    non_bg_samples += non_bg_mask.sum().item()
            avg_val_loss = total_val_loss / len(val_loader)
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            val_accuracy = (all_preds == all_targets).sum() / all_targets.size
            val_f1 = f1_score(all_targets.flatten(), all_preds.flatten(), average='macro', zero_division=0)
            non_bg_accuracy = non_bg_correct / max(non_bg_samples, 1)
        except Exception as e:
            logger.error(f"Error during validation epoch {epoch+1}: {str(e)}")
            raise

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CONFIG["training"]["model_save_path"])
            logger.info(f"Model saved at {CONFIG['training']['model_save_path']} (Best Val F1: {val_f1:.6f})")

        lr_history.append(optimizer.param_groups[0]['lr'])
        val_acc_history.append(val_accuracy)
        non_bg_acc_history.append(non_bg_accuracy)
        val_f1_history.append(val_f1)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                    f"Val Accuracy: {val_accuracy:.6f}, Val F1: {val_f1:.6f}, Non-BG Accuracy: {non_bg_accuracy:.6f}, "
                    f"Learning Rate: {optimizer.param_groups[0]['lr']:.6e}")

    plot_lr_schedule(lr_history, num_epochs, CONFIG["training"]["results_dir"])
    plot_training_metrics(val_acc_history, non_bg_acc_history, val_f1_history, train_loss_history, val_loss_history, num_epochs, CONFIG["training"]["results_dir"])

    return {
        "lr_history": lr_history,
        "val_acc_history": val_acc_history,
        "non_bg_acc_history": non_bg_acc_history,
        "val_f1_history": val_f1_history,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
    }

def main():
    torch.set_num_threads(16)
    logger.info(f"Starting training at {datetime.now()}")
    
    os.makedirs(CONFIG["training"]['results_dir'], exist_ok=True)
    
    try:
        with open(CONFIG["data"]["element_map_path"], 'r') as f:
            element_map = json.load(f)
        logger.info(f"Loaded element map: {list(element_map.keys())}")
        logger.info(f"Number of elements in dataset: {len(element_map)}")
        logger.info(f"Filtered to required elements: {REQUIRED_ELEMENTS}")
    except Exception as e:
        logger.error(f"Failed to load element_map: {str(e)}")
        return

    try:
        train_spectrum_dataset = HDF5SpectrumDataset(
            CONFIG["data"]["dataset_path"],
            CONFIG["data"]["train_split"],
            element_map,
            REQUIRED_ELEMENTS,
            CONFIG["data"]["max_train_samples"],
            augment_noise_std=CONFIG["data"]["augment_noise_std"]
        )
        val_spectrum_dataset = HDF5SpectrumDataset(
            CONFIG["data"]["dataset_path"],
            CONFIG["data"]["val_split"],
            element_map,
            REQUIRED_ELEMENTS,
            CONFIG["data"]["max_val_samples"],
            augment_noise_std=0.0
        )
        logger.info(f"Train dataset loaded, num_classes: {train_spectrum_dataset.num_classes}")
        logger.info(f"Validation dataset loaded, num_classes: {val_spectrum_dataset.num_classes}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return

    train_data, train_targets = train_spectrum_dataset.get_data()
    val_data, val_targets = val_spectrum_dataset.get_data()
    if len(train_data) == 0 or len(val_data) == 0:
        logger.error("No valid samples found in train or validation dataset")
        return

    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_targets))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_targets))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True, num_workers=2)  # Kurangi num_workers
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=False, num_workers=2)

    try:
        model = InformerModel(
            input_dim=CONFIG["model"]["input_dim"],
            d_model=CONFIG["model"]["d_model"],
            nhead=CONFIG["model"]["nhead"],
            num_encoder_layers=CONFIG["model"]["num_encoder_layers"],
            dim_feedforward=CONFIG["model"]["dim_feedforward"],
            dropout=CONFIG["model"]["dropout"],
            seq_length=CONFIG["model"]["seq_length"],
            num_classes=train_spectrum_dataset.num_classes,
            attn_factor=CONFIG["model"]["attn_factor"]
        ).to(CONFIG["training"]["device"])
        logger.info(f"Informer model initialized with num_classes: {train_spectrum_dataset.num_classes}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return

    summarize_model(model, CONFIG, CONFIG["training"]["results_dir"])

    try:
        class_weights = compute_class_weights(
            train_targets.argmax(-1),
            train_spectrum_dataset.num_classes,
            CONFIG["training"]["class_weight_factor"]
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(CONFIG["training"]["device"]))
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], weight_decay=1e-5)
        # Nonaktifkan IPEX sementara untuk menghindari masalah autograd
        # if ipex:
        #     logger.info("Applying IPEX optimization to model and optimizer")
        #     model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        # else:
        #     logger.warning("IPEX not available, skipping optimization")
        torch.save(class_weights, CONFIG["training"]["class_weight_path"])
        logger.info(f"Class weights saved to {CONFIG['training']['class_weight_path']}")
    except Exception as e:
        logger.error(f"Failed to initialize criterion or optimizer: {str(e)}")
        return

    train_model(model, train_loader, val_loader, criterion, optimizer, CONFIG["training"]["num_epochs"], CONFIG["training"]["device"])

if __name__ == "__main__":
    main()