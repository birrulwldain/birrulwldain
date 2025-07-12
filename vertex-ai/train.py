import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import json
import os
import random
import argparse
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import math
import gc
import time
from torch.amp import GradScaler, autocast
import textwrap

# Mengatur variabel lingkungan untuk mencegah fragmentasi memori di PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- KONFIGURASI UTAMA (Path akan diisi secara dinamis) ---
CONFIG = {
    "model": {
        "input_dim": 1, 
        "d_model": 32, 
        "nhead": 4, 
        "num_encoder_layers": 2,
        "dim_feedforward": 64, 
        "dropout": 0.4, 
        "seq_length": 4096,
        "attn_factor": 5, 
        "num_classes": 18
    },
    "training": {
        "batch_size": 25, 
        "num_epochs": 10, 
        "learning_rate": 1e-7,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "weight_decay": 1e-4, 
        "gradient_accumulation_steps": 4,
        "early_stopping_patience": 3
    }
}

# --- KELAS DATASET ---
class HDF5SpectrumDataset(torch.utils.data.Dataset):
    """Dataset HDF5 efisien yang hanya memuat data saat diperlukan (Lazy Loading)."""
    def __init__(self, file_path, split, max_samples=None):
        self.file_path = file_path
        self.split = split
        self.h5_file = None
        with h5py.File(self.file_path, 'r') as f:
            if self.split not in f:
                raise KeyError(f"Split '{self.split}' tidak ditemukan di file HDF5.")
            total_len = len(f[self.split]['spectra'])
            self.dataset_len = min(total_len, max_samples) if max_samples is not None else total_len
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Dataset '{self.split}' diinisialisasi dengan {self.dataset_len} sampel.")
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
        spectra = self.h5_file[self.split]['spectra'][idx][..., np.newaxis].astype(np.float32)
        labels = self.h5_file[self.split]['labels'][idx].astype(np.float32)
        return torch.from_numpy(spectra), torch.from_numpy(labels)

# --- FUNGSI UTILITAS ---
def compute_class_weights(dataset, num_classes, batch_size=256, num_workers=2):
    """Menghitung bobot POSITIF untuk setiap kelas dalam konteks multi-label."""
    class_pos_counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0
    temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Menghitung class weights (multi-label)...")
    for _, labels in tqdm(temp_loader, desc="Calculating Weights"):
        class_pos_counts += labels.sum(dim=(0, 1)).cpu().numpy()
        total_pixels += labels.numel() / num_classes
        
    weights = total_pixels / (class_pos_counts + 1e-6)
    weights = np.clip(weights, 1.0, 50.0)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Perhitungan Class weights selesai.")
    return torch.tensor(weights, dtype=torch.float32)

def plot_and_save_history(history, save_dir):
    """Membuat dan menyimpan plot dari histori training."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':')
    
    # Plot Metrik
    ax2.plot(epochs, history['val_f1'], 'go-', label='Validation F1-Score (Samples)')
    ax2.set_title('Validation F1-Score & Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1-Score')
    ax2.grid(True, linestyle=':')
    
    ax2b = ax2.twinx()
    ax2b.plot(epochs, history['val_acc'], 'ms--', label='Validation Accuracy (Subset)', alpha=0.6)
    ax2b.set_ylabel('Accuracy (Subset)', color='m')
    ax2b.tick_params(axis='y', labelcolor='m')
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history_plot.png")
    plt.savefig(save_path)
    print(f"\nPlot histori training disimpan di: {save_path}")
    plt.close(fig)

def set_seed(seed_value=42):
    """Mengatur seed untuk hasil yang konsisten."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- ARSITEKTUR MODEL ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, _ = x.size()
        Q = self.q_linear(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        u = min(L, int(self.factor * math.log(L)) if L > 1 else L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        top_k, _ = torch.topk(scores, u, dim=-1)
        mask_topk = scores < top_k[..., -1].unsqueeze(-1)
        scores = scores.masked_fill(mask_topk, -1e4)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_linear(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, attn_factor=5):
        super(EncoderLayer, self).__init__()
        self.self_attention = ProbSparseSelfAttention(d_model, nhead, dropout, attn_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x

class InformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, seq_length, num_classes, attn_factor):
        super(InformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout, attn_factor) for _ in range(num_encoder_layers)])
        self.decoder = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.decoder(x)

    def train_model(self, train_loader, val_loader, criterion, optimizer, scheduler, config):
        scaler = GradScaler()
        histories = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        best_val_f1 = 0.0
        epochs_no_improve = 0
        device = config['training']['device']
        num_epochs = config['training']['num_epochs']
        patience = config['training']['early_stopping_patience']
        grad_accum = config['training']['gradient_accumulation_steps']
        model_path = config['training']['model_save_path']

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
            for i, (data, labels) in enumerate(pbar):
                data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast(device_type=device, dtype=torch.float16):
                    output = self(data)
                    loss = criterion(output, target) / grad_accum
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                train_loss += loss.item() * grad_accum
                pbar.set_postfix({'Loss': f'{train_loss / (i + 1):.4f}'})
            histories['train_loss'].append(train_loss / len(train_loader))

            self.eval()
            val_loss = 0.0
            all_preds_flat, all_targets_flat = [], []
            with torch.no_grad():
                for data, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}-Validation", leave=False):
                    data, target = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast(device_type=device, dtype=torch.float16):
                        output = self(data)
                        loss = criterion(output, target)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(output) > 0.5).float()
                    all_preds_flat.append(preds.view(-1, self.num_classes).cpu())
                    all_targets_flat.append(target.view(-1, self.num_classes).cpu())
            histories['val_loss'].append(val_loss / len(val_loader))
            
            all_preds = torch.cat(all_preds_flat).numpy()
            all_targets = torch.cat(all_targets_flat).numpy()
            val_acc = accuracy_score(all_targets, all_preds)
            val_f1 = f1_score(all_targets, all_preds, average='samples', zero_division=0)
            histories['val_acc'].append(val_acc)
            histories['val_f1'].append(val_f1)
            
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Epoch {epoch+1}/{num_epochs} | Train Loss: {histories['train_loss'][-1]:.4f} | Val Loss: {histories['val_loss'][-1]:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.state_dict(), model_path)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Model terbaik disimpan dengan F1: {val_f1:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if scheduler:
                scheduler.step(val_f1)
            
            if epochs_no_improve >= patience:
                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Early stopping dipicu.")
                break
        return histories

# --- FUNGSI LOSS ---
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_loss = ((1 - pt)**self.gamma * bce_loss)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# --- FUNGSI PARSE ARGUMEN ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Informer Multi-Label Training on HPC")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path ke file dataset HDF5.')
    parser.add_argument('--element_map_path', type=str, required=True, help='Path ke file peta elemen JSON.')
    parser.add_argument('--model_dir', type=str, required=True, help='Direktori untuk menyimpan file model (.pth).')
    parser.add_argument('--results_dir', type=str, required=True, help='Direktori untuk menyimpan hasil (plot, log, dll.).')
    parser.add_argument('--epochs', type=int, default=CONFIG['training']['num_epochs'], help='Jumlah epoch training.')
    parser.add_argument('--batch_size', type=int, default=CONFIG['training']['batch_size'], help='Ukuran batch.')
    parser.add_argument('--lr', type=float, default=CONFIG['training']['learning_rate'], help='Learning rate.')
    return parser.parse_args()

# --- FUNGSI UTAMA (MAIN) ---
def main(args):
    set_seed(42)
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    CONFIG['training']['model_save_path'] = os.path.join(args.model_dir, 'informer_multilabel_model.pth')
    CONFIG['training']['class_weight_path'] = os.path.join(args.model_dir, 'class_weights_multilabel.pth')
    CONFIG['training']['num_epochs'] = args.epochs
    CONFIG['training']['batch_size'] = args.batch_size
    CONFIG['training']['learning_rate'] = args.lr

    device = CONFIG["training"]["device"]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING JOB ---")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device.upper()}")
    
    dl_args = {'num_workers': 2, 'pin_memory': True}
    train_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="train")
    val_dataset = HDF5SpectrumDataset(file_path=args.dataset_path, split="validation")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True, **dl_args)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["training"]["batch_size"] * 2, shuffle=False, **dl_args)

    model = InformerModel(**CONFIG["model"]).to(device)

    model_path = CONFIG["training"]["model_save_path"]
    if os.path.exists(model_path):
        print(f"\nModel ditemukan. Memuat bobot dari: {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Gagal memuat model, memulai dari awal. Error: {e}")
    else:
        print(f"\nTidak ada model yang ditemukan. Memulai training dari awal.")

    if int(torch.__version__.split('.')[0]) >= 2:
        print(f"Mengompilasi model dengan torch.compile()...")
        model = torch.compile(model)

    class_weight_path = CONFIG["training"]["class_weight_path"]
    if os.path.exists(class_weight_path):
        print(f"Memuat class weights dari: {class_weight_path}")
        class_weights = torch.load(class_weight_path).to(device)
    else:
        print(f"Menghitung class weights baru...")
        class_weights = compute_class_weights(train_dataset, CONFIG["model"]["num_classes"], num_workers=dl_args['num_workers']).to(device)
        torch.save(class_weights, class_weight_path)
        print(f"Class weights disimpan di: {class_weight_path}")

    criterion = MultiLabelFocalLoss(pos_weight=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["training"]["learning_rate"], weight_decay=CONFIG["training"]["weight_decay"], fused=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.02, patience=2, verbose=True)

    histories = model.train_model(
        train_loader, val_loader, criterion, optimizer, scheduler, CONFIG
    )

    if histories:
        plot_and_save_history(histories, args.results_dir)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- JOB FINISHED ---")

# --- TITIK MASUK EKSEKUSI ---
if __name__ == "__main__":
    args = parse_arguments()
    main(args)

