import argparse
import json
import pathlib
import hashlib
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer


class PromptEmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class BottleneckAE(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, bottleneck_dim),
        )
        self.z_norm = nn.LayerNorm(bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        x = self.input_norm(x)
        z = self.encoder(x)
        z = self.z_norm(z)
        recon = self.decoder(z)
        preds = self.reg_head(z)
        return recon, preds, z


def load_cfg(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_output_suffix(cfg: Dict[str, Any]) -> Dict[str, Any]:
    suffix = str(cfg.get("output_suffix", "")).strip()
    if not suffix:
        return cfg
    base = str(cfg.get("output_dir", "output"))
    if not base.endswith(f"_{suffix}") and not base.endswith(f"/{suffix}"):
        base = f"{base}_{suffix}"
    cfg["output_dir"] = base
    for key in ("prompts_file", "embedding_cache", "ae_checkpoint"):
        path = cfg.get(key)
        if isinstance(path, str) and path.startswith("output/"):
            cfg[key] = path.replace("output/", f"{base}/", 1)
    return cfg


def resolve_snapshot_path(base_path: pathlib.Path) -> pathlib.Path:
    snap_root = base_path / "snapshots"
    if snap_root.exists():
        snaps = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for s in snaps:
            if (s / "config.json").exists():
                return s
    return base_path


def normalize_model_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(cfg)
    for key in ("model_path", "tokenizer_path"):
        raw = updated.get(key)
        if not raw:
            continue
        resolved = resolve_snapshot_path(pathlib.Path(raw))
        updated[key] = str(resolved)
    return updated


def masked_mean(hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    attn = attn.unsqueeze(-1)
    summed = (hidden * attn).sum(dim=1)
    denom = attn.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def compute_embeddings(
    prompts: List[Tuple[str, List[float]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    model.eval()
    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for text, label in prompts:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=True,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens, output_hidden_states=True)
            penultimate_hidden = out.hidden_states[-2]
            attn = tokens["attention_mask"]
            last_idx = attn.sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(penultimate_hidden.size(0), device=penultimate_hidden.device)
            last_token = penultimate_hidden[batch_idx, last_idx]
            all_embeds.append(last_token.squeeze(0).cpu())
            all_labels.append(torch.tensor(label, dtype=torch.float32))

    embeddings = torch.stack(all_embeds, dim=0)
    labels = torch.stack(all_labels, dim=0)
    return embeddings, labels


def load_prompts(prompts_file: pathlib.Path) -> List[Tuple[str, List[float]]]:
    pairs = []
    with prompts_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            persona = record["persona"]
            label = [
                float(persona["Honesty-Humility"]),
                float(persona["Emotionality"]),
                float(persona["Extraversion"]),
                float(persona["Agreeableness"]),
                float(persona["Conscientiousness"]),
                float(persona["Openness"]),
            ]
            for p in record["prompts"]:
                pairs.append((p, label))
    return pairs


def compute_file_sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_cache(
    path: pathlib.Path,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    meta: Optional[Dict[str, Any]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "labels": labels, "meta": meta or {}}, path)


def load_cache(path: pathlib.Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    return obj["embeddings"], obj["labels"], obj.get("meta", {})


def eval_loop(model, loader, recon_loss_fn, sup_loss_fn, lam, device):
    model.eval()
    target_dtype = next(model.parameters()).dtype
    tot_recon = tot_sup = tot = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=target_dtype)
            yb = yb.to(device, dtype=target_dtype)
            recon, preds, _ = model(xb)
            recon_loss = recon_loss_fn(recon, xb)
            sup_loss = sup_loss_fn(preds, yb)
            loss = recon_loss + lam * sup_loss
            tot_recon += recon_loss.item()
            tot_sup += sup_loss.item()
            tot += loss.item()
    n = max(1, len(loader))
    return tot_recon / n, tot_sup / n, tot / n


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    recon_loss_fn,
    sup_loss_fn,
    lam,
    device,
    epochs,
    grad_clip,
    scheduler,
    ckpt_path: pathlib.Path,
    early_stop: int,
):
    best_val = float("inf")
    patience = 0
    model.train()
    target_dtype = next(model.parameters()).dtype
    for ep in range(epochs):
        tot_recon = tot_sup = tot = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, dtype=target_dtype)
            yb = yb.to(device, dtype=target_dtype)
            recon, preds, _ = model(xb)
            recon_loss = recon_loss_fn(recon, xb)
            sup_loss = sup_loss_fn(preds, yb)
            loss = recon_loss + lam * sup_loss
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tot_recon += recon_loss.item()
            tot_sup += sup_loss.item()
            tot += loss.item()

        n = max(1, len(train_loader))
        train_recon = tot_recon / n
        train_sup = tot_sup / n
        train_total = tot / n

        val_recon, val_sup, val_total = eval_loop(
            model, val_loader, recon_loss_fn, sup_loss_fn, lam, device
        )

        if scheduler is not None:
            scheduler.step(val_total)

        print(
            f"Epoch {ep+1}/{epochs} | "
            f"Train Recon {train_recon:.4f} | Train Sup {train_sup:.4f} | Train Total {train_total:.4f} | "
            f"Val Recon {val_recon:.4f} | Val Sup {val_sup:.4f} | Val Total {val_total:.4f}"
        )

        if val_total < best_val:
            best_val = val_total
            patience = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if early_stop and patience >= early_stop:
                print(f"early stopping at epoch {ep+1}")
                break


def main():
    default_cfg = pathlib.Path("config/ae_deepseek-r1-distill-qwen-7b.json")
    if not default_cfg.exists():
        default_cfg = pathlib.Path("config/ae.json")
    parser = argparse.ArgumentParser(description="Train bottleneck autoencoder")
    parser.add_argument(
        "--config",
        default=str(default_cfg),
        help="path to AE config json",
    )
    args = parser.parse_args()
    cfg = apply_output_suffix(load_cfg(pathlib.Path(args.config)))
    cfg = normalize_model_paths(cfg)
    torch.manual_seed(int(cfg.get("seed", 42)))

    model_base = pathlib.Path(cfg["model_path"])
    tok_base = pathlib.Path(cfg.get("tokenizer_path", cfg["model_path"]))

    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_base),
        use_fast=True,
        legacy=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    llama = AutoModelForCausalLM.from_pretrained(
        str(model_base),
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=True,
        output_hidden_states=True,
    )

    prompts_file = pathlib.Path(cfg["prompts_file"])
    cache_path = pathlib.Path(cfg["embedding_cache"])

    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"prompts file not found: {prompts_file}. "
            "Generate prompts first or update prompts_file in the config."
        )

    prompts_hash = compute_file_sha1(prompts_file)

    if cache_path.exists():
        embeddings, labels, meta = load_cache(cache_path)
        cache_hash = meta.get("prompts_sha1")
        cache_n = meta.get("num_prompts")
        if cache_hash != prompts_hash or (cache_n is not None and cache_n != embeddings.size(0)):
            print("cache mismatch detected, recomputing embeddings")
            pairs = load_prompts(prompts_file)
            embeddings, labels = compute_embeddings(pairs, tokenizer, llama, cfg.get("max_seq_len", 256))
            meta = {
                "prompts_sha1": prompts_hash,
                "num_prompts": len(pairs),
                "model_path": str(model_base),
                "tokenizer_path": str(tok_base),
                "max_seq_len": int(cfg.get("max_seq_len", 256)),
            }
            save_cache(cache_path, embeddings, labels, meta)
            print(f"saved embeddings to {cache_path}")
        else:
            print(f"loaded cached embeddings from {cache_path}")
    else:
        pairs = load_prompts(prompts_file)
        embeddings, labels = compute_embeddings(pairs, tokenizer, llama, cfg.get("max_seq_len", 256))
        meta = {
            "prompts_sha1": prompts_hash,
            "num_prompts": len(pairs),
            "model_path": str(model_base),
            "tokenizer_path": str(tok_base),
            "max_seq_len": int(cfg.get("max_seq_len", 256)),
        }
        save_cache(cache_path, embeddings, labels, meta)
        print(f"saved embeddings to {cache_path}")

    embeddings = embeddings.float()
    labels = labels.float()
    input_dim = embeddings.size(1)
    dataset = PromptEmbeddingDataset(embeddings, labels)
    val_split = float(cfg.get("val_split", 0.1))
    val_size = max(1, int(len(dataset) * val_split))
    train_size = max(1, len(dataset) - val_size)
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.get("batch_size", 32), shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ae = BottleneckAE(
        input_dim=input_dim,
        bottleneck_dim=cfg.get("bottleneck_dim", 6),
        dropout_p=float(cfg.get("dropout", 0.0)),
    ).to(device)
    recon_loss_fn = nn.MSELoss()
    sup_loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        ae.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("lr_decay", 0.5)),
        patience=int(cfg.get("lr_patience", 3)),
    )

    ckpt_path = pathlib.Path(cfg.get("ae_checkpoint", "output/ae_bottleneck.pt"))
    train_loop(
        ae,
        train_loader,
        val_loader,
        optimizer,
        recon_loss_fn,
        sup_loss_fn,
        cfg.get("lambda_supervise", 1.0),
        device,
        cfg.get("epochs", 20),
        float(cfg.get("grad_clip", 1.0)),
        scheduler,
        ckpt_path,
        int(cfg.get("early_stop", 0)),
    )

    print(f"saved best AE to {ckpt_path}")


if __name__ == "__main__":
    main()
