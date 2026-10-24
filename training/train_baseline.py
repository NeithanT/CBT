import os
import sys
import argparse
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.model import MathGPT, GPTConfig
from src.tokenizer import MathTokenizer
from src.dataset import create_dataloaders
from src.constants import TRAINING_CONSTANTS, MODEL_CONSTANTS, get_device

DEFAULT_SAVE_PATH = os.path.join(ROOT_DIR, "checkpoints", "toy_baselines", "v25_abspos_3l96d.pt")


def train(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    n_train: int = 10000,
    n_val: int = 1000,
    mask_prompt: bool = True,
    save_path: str = DEFAULT_SAVE_PATH,
    log_interval: int = 50,
    device: Optional[torch.device] = None,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if device is None:
        device = get_device()

    print(f"Training Baseline MathGPT on {device}...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        num_train=n_train,
        num_val=n_val,
        batch_size=batch_size,
        block_size=MODEL_CONSTANTS.block_size,
        mask_prompt=mask_prompt,
    )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=MODEL_CONSTANTS.block_size,
        n_layer=MODEL_CONSTANTS.n_layer,
        n_head=MODEL_CONSTANTS.n_head,
        n_embd=MODEL_CONSTANTS.n_embd,
        dropout=MODEL_CONSTANTS.dropout,
        pos_emb_type=MODEL_CONSTANTS.pos_emb_type,
    )

    model = MathGPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{step + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, vloss = model(x_val, y_val)
                val_loss += vloss.item()
        val_loss /= len(val_loader)
        print(f"--> Epoch {epoch} Validation Loss: {val_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_size": tokenizer.vocab_size,
        },
        save_path,
    )
    print(f"Baseline saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline MathGPT Trainer")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONSTANTS.epochs)
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONSTANTS.batch_size)
    parser.add_argument("--lr", type=float, default=TRAINING_CONSTANTS.lr)
    parser.add_argument("--n_train", type=int, default=TRAINING_CONSTANTS.n_train)
    parser.add_argument("--n_val", type=int, default=TRAINING_CONSTANTS.n_val)
    parser.add_argument("--mask_prompt", action="store_true", default=TRAINING_CONSTANTS.mask_prompt)
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--log_interval", type=int, default=TRAINING_CONSTANTS.log_interval)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_train=args.n_train,
        n_val=args.n_val,
        mask_prompt=args.mask_prompt,
        save_path=args.save_path,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
