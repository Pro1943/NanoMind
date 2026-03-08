import torch
import time
import os
from config import BATCH_SIZE, BLOCK_SIZE, MAX_ITERS, EVAL_EVERY, LEARN_RATE, DEVICE, WEIGHTS_FILE

def get_batch(data, split, train_data, val_data):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i:i+BLOCK_SIZE]     for i in ix])
    y   = torch.stack([src[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        L = torch.zeros(10)
        for k in range(10):
            x, y = get_batch(None, split, train_data, val_data)
            _, loss = model(x, y)
            L[k] = loss.item()
        out[split] = L.mean().item()
    model.train()
    return out

def train(model):
    os.makedirs("model", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=1e-5)

    data       = model._train_data
    train_data = data[:int(0.9 * len(data))]
    val_data   = data[int(0.9 * len(data)):]

    print("Training...")
    print("─" * 55)

    t0       = time.time()
    best_val = float("inf")

    for step in range(MAX_ITERS + 1):
        if step % EVAL_EVERY == 0:
            losses  = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            pct     = step / MAX_ITERS
            bar     = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
            eta_str = "..." if step == 0 else f"{(elapsed / step) * (MAX_ITERS - step) / 60:.1f}m"
            print(f"[{bar}] {int(pct*100):3d}% | loss {losses['train']:.3f} | val {losses['val']:.3f} | ETA {eta_str}")

            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(model.state_dict(), WEIGHTS_FILE)

        if step == MAX_ITERS:
            break

        x, y = get_batch(None, "train", train_data, val_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print(f"\nDone in {time.time()-t0:.0f}s  |  best val loss: {best_val:.3f}")
    print(f"Weights saved → {WEIGHTS_FILE}")