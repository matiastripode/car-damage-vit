import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score


def entrenar(modelo, train_loader, val_loader, config, epoch_callback=None):
    """
    Loop principal de entrenamiento con validación por epoch.

    Args:
        modelo:       modelo HuggingFace (AutoModelForImageClassification)
        train_loader: DataLoader de entrenamiento (con WeightedRandomSampler)
        val_loader:   DataLoader de validación
        config:       dict con claves:
                        device      (str, default "cpu")
                        lr          (float, default 2e-4)
                        epochs      (int, default 20)
                        patience    (int, default 5)
                        output_dir  (str, default "checkpoints")

    Returns:
        history: lista de dicts por epoch con train_loss, val_loss, val_f1, val_acc
    """
    device     = torch.device(config.get("device", "cpu"))
    lr         = config.get("lr", 2e-4)
    epochs     = config.get("epochs", 20)
    patience   = config.get("patience", 5)
    output_dir = Path(config.get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    modelo = modelo.to(device)
    # weight_decay=1e-3 (vs 1e-4 original) para penalizar pesos grandes y reducir overfitting.
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-3)
    # ReduceLROnPlateau baja el LR solo cuando val_f1 deja de mejorar,
    # a diferencia de CosineAnnealingLR que sigue un schedule fijo
    # sin importar si el modelo mejora o no.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best_f1    = 0.0
    sin_mejora = 0
    history    = []

    for epoch in range(epochs):
        t0 = time.perf_counter()

        # ── Entrenamiento ────────────────────────────────────────────────────
        modelo.train()
        train_loss = 0.0
        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)
            optimizer.zero_grad()
            out = modelo(pixel_values=pixel_values, labels=labels)
            out.loss.backward()
            optimizer.step()
            train_loss += out.loss.item()
        train_loss /= len(train_loader)

        # ── Validación ───────────────────────────────────────────────────────
        modelo.eval()
        val_loss   = 0.0
        preds_all  = []
        labels_all = []
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels       = labels.to(device)
                out = modelo(pixel_values=pixel_values, labels=labels)
                val_loss += out.loss.item()
                preds_all.extend(out.logits.argmax(dim=-1).cpu().tolist())
                labels_all.extend(labels.cpu().tolist())
        val_loss /= len(val_loader)

        f1  = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        acc = accuracy_score(labels_all, preds_all)
        dt  = time.perf_counter() - t0
        # Ajusta el LR basándose en val_f1 real, no en un schedule predefinido.
        scheduler.step(f1)

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_f1":     f1,
            "val_acc":    acc,
        })
        if epoch_callback is not None:
            epoch_callback(history[-1])

        print(
            f"[{epoch:02d}] train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={f1:.4f} | "
            f"val_acc={acc:.4f} | {dt:.1f}s"
        )

        # ── Checkpoint + early stopping ──────────────────────────────────────
        if f1 > best_f1 + 1e-4:
            best_f1    = f1
            sin_mejora = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": modelo.state_dict(),
                    "val_f1":           f1,
                    "history":          history,
                },
                output_dir / "best_model.pt",
            )
            print(f"  → nuevo mejor checkpoint (val_f1={f1:.4f})")
        else:
            sin_mejora += 1
            print(f"  → sin mejora ({sin_mejora}/{patience})")
            if sin_mejora >= patience:
                print("Early stopping activado.")
                break

    return history
