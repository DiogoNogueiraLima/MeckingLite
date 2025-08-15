# scripts/tune_supervised.py
import os, sys, yaml, math, csv, random, time
import optuna
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# ==== NOVO: matplotlib para salvar PNGs (sem abrir janela) ====
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train_supervised import (
    load_config,
    make_datasets_and_loaders,
    make_model,
    run_one_epoch,
    save_ckpt,
    latest_checkpoint,
)


# ==== NOVO: helpers de diretório e plots ====
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _plot_metric_pair(ax, epochs, tr_vals, va_vals, title, ylabel):
    ax.plot(epochs, tr_vals, label="train")
    ax.plot(epochs, va_vals, label="val")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def save_png_plots(run_name: str, hist: dict, out_dir: str):
    """
    Salva PNGs com curvas de métricas.
    hist: {
       'epoch': [1,2,...],
       'train': {'loss':[], 'top1':[], 'top3':[], 'mrr10':[], 'ece':[], 'entropy':[]},
       'val':   {'loss':[], 'top1':[], 'top3':[], 'mrr10':[], 'ece':[], 'entropy':[]},
    }
    """
    _ensure_dir(out_dir)
    epochs = hist["epoch"]
    tr = hist["train"]
    va = hist["val"]
    if len(epochs) == 0:
        return

    # 1) Figura consolidada (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _plot_metric_pair(axes[0, 0], epochs, tr["loss"], va["loss"], "Loss", "loss")
    _plot_metric_pair(axes[0, 1], epochs, tr["top1"], va["top1"], "Top-1", "acc")
    _plot_metric_pair(axes[0, 2], epochs, tr["top3"], va["top3"], "Top-3", "acc")
    _plot_metric_pair(axes[1, 0], epochs, tr["mrr10"], va["mrr10"], "MRR@10", "score")
    _plot_metric_pair(axes[1, 1], epochs, tr["ece"], va["ece"], "ECE", "abs err")
    _plot_metric_pair(
        axes[1, 2], epochs, tr["entropy"], va["entropy"], "Entropia", "nats"
    )
    fig.suptitle(f"Evolução — {run_name}", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{run_name}_summary.png"), dpi=150)
    plt.close(fig)

    # 2) PNGs individuais por métrica
    for key, ylabel in [
        ("loss", "loss"),
        ("top1", "acc"),
        ("top3", "acc"),
        ("mrr10", "score"),
        ("ece", "abs err"),
        ("entropy", "nats"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        _plot_metric_pair(ax, epochs, tr[key], va[key], key.upper(), ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{run_name}_{key}.png"), dpi=150)
        plt.close(fig)


# ==== loader com subset ====
def get_subset_loader(loader, frac=0.1):
    ds = loader.dataset
    n = len(ds)
    m = max(1000, int(n * frac))  # pelo menos 1000 amostras
    idxs = random.sample(range(n), k=min(m, n))
    subset = Subset(ds, idxs)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


def objective(
    trial, cfg, phase, schedule, num_moves, base_train_loader, base_val_loader
):
    # espaços de busca
    space = cfg["optuna"]["search_space"]
    lr = trial.suggest_categorical("lr", space["lr"])
    wd = trial.suggest_categorical("weight_decay", space["weight_decay"])
    dropout = trial.suggest_categorical("dropout", space["dropout"])
    bs = trial.suggest_categorical("batch_size", space["batch_size"])
    accum_steps = trial.suggest_categorical("accum_steps", space["accum_steps"])
    label_smoothing = 0.0
    if cfg.get("mode", "top3") == "top1":
        label_smoothing = trial.suggest_categorical(
            "label_smoothing", space["label_smoothing"]
        )

    # subset loaders
    sample_frac = float(cfg["optuna"].get("sample_fraction", 0.12))
    train_loader = get_subset_loader(base_train_loader, frac=sample_frac)
    val_loader = get_subset_loader(base_val_loader, frac=min(0.10, sample_frac))

    # modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("use_amp", True))
    net_cfg = dict(cfg)
    net_cfg["dropout"] = float(dropout)
    net_cfg["batch_size"] = int(bs)  # informativo
    net_cfg["accum_steps"] = int(accum_steps)  # informativo
    net = (
        make_model(net_cfg, num_moves).to(device).to(memory_format=torch.channels_last)
    )

    # otim
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=float(lr), weight_decay=float(wd)
    )
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mode = cfg.get("mode", "top3")

    # logging por trial
    tb_root = cfg["logging"].get("tensorboard_dir", "runs")
    tb_dir = os.path.join(tb_root, f"{phase}_trials", f"trial_{trial.number:04d}")
    writer = SummaryWriter(log_dir=tb_dir)
    csv_dir = cfg["logging"].get("csv_dir", "logs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{phase}_trial_{trial.number:04d}.csv")

    # checkpoints por trial
    ckpt_dir = os.path.join(
        cfg.get("output_dir", "checkpoints"),
        "trials",
        phase,
        f"trial_{trial.number:04d}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # ==== NOVO: histórico para PNGs por trial ====
    run_name = f"{phase}_trial{trial.number:04d}"
    png_dir = os.path.join(csv_dir, "png", run_name)
    _ensure_dir(png_dir)
    hist = {
        "epoch": [],
        "train": {k: [] for k in ["loss", "top1", "top3", "mrr10", "ece", "entropy"]},
        "val": {k: [] for k in ["loss", "top1", "top3", "mrr10", "ece", "entropy"]},
    }

    max_epochs = int(cfg["optuna"].get("max_epochs_per_trial", 2))
    best_val = float("inf")
    patience = 2
    bad = 0

    step = 0
    for ep in range(1, max_epochs + 1):
        tr = run_one_epoch(
            net,
            train_loader,
            device,
            use_amp,
            criterion_kl if mode == "top3" else criterion_ce,
            mode,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=float(cfg.get("grad_clip", 0.0)),
        )
        va = run_one_epoch(
            net,
            val_loader,
            device,
            use_amp,
            criterion_kl if mode == "top3" else criterion_ce,
            mode,
            optimizer=None,
            scaler=None,
        )

        # logs (TensorBoard)
        for k, v in tr.items():
            writer.add_scalar(f"train/{k}", v, step)
        for k, v in va.items():
            writer.add_scalar(f"val/{k}", v, step)

        # logs (CSV)
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    ["trial", "epoch", "step"]
                    + [f"train_{k}" for k in tr.keys()]
                    + [f"val_{k}" for k in va.keys()]
                    + [
                        "lr",
                        "wd",
                        "dropout",
                        "batch_size",
                        "accum_steps",
                        "label_smoothing",
                    ]
                )
            w.writerow(
                [trial.number, ep, step]
                + list(tr.values())
                + list(va.values())
                + [lr, wd, dropout, bs, accum_steps, label_smoothing]
            )

        # checkpoint
        meta = {
            "phase": phase,
            "trial": trial.number,
            "epoch": ep,
            "hparams": {
                "lr": lr,
                "wd": wd,
                "dropout": dropout,
                "batch_size": bs,
                "accum_steps": accum_steps,
                "label_smoothing": label_smoothing,
            },
        }
        ckpt_path = os.path.join(ckpt_dir, f"ep{ep:02d}.ckpt")
        save_ckpt(ckpt_path, net, optimizer, scaler, meta)

        # ==== NOVO: atualizar histórico e salvar PNGs do trial ====
        hist["epoch"].append(ep)
        for k in hist["train"].keys():
            hist["train"][k].append(tr[k])
            hist["val"][k].append(va[k])
        save_png_plots(run_name, hist, png_dir)

        # reporta para Optuna
        trial.report(va["loss"], ep)
        if va["loss"] < best_val - 1e-6:
            best_val = va["loss"]
            bad = 0
            save_ckpt(
                os.path.join(ckpt_dir, f"best.ckpt"), net, optimizer, scaler, meta
            )
        else:
            bad += 1
            if trial.should_prune() or bad >= patience:
                writer.close()
                raise optuna.TrialPruned()

        step += 1

    writer.close()
    return best_val


def main():
    cfg = load_config()
    if not cfg.get("optuna", {}).get("enable", False):
        print("Optuna desativado no config.")
        return

    phase = cfg.get("train_phase", "v1_depth6")
    schedule = cfg["curriculum"][f"phase_{phase}"]["sample_weights"]

    # base loaders (vamos só amostrar subsets dentro do objective)
    _, _, base_train_loader, base_val_loader, num_moves = make_datasets_and_loaders(
        cfg, schedule["start"], batch_size=cfg.get("batch_size", 256), workers=2
    )

    storage = cfg["optuna"].get("storage", None)
    study_name = cfg["optuna"].get("study_name", "policy_tuning")
    direction = cfg["optuna"].get("direction", "minimize")

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True, direction=direction
    )

    n_trials = int(cfg["optuna"].get("n_trials", 20))
    timeout_m = int(cfg["optuna"].get("timeout_minutes", 0))
    timeout_s = timeout_m * 60 if timeout_m > 0 else None

    try:
        study.optimize(
            lambda tr: objective(
                tr, cfg, phase, schedule, num_moves, base_train_loader, base_val_loader
            ),
            n_trials=n_trials,
            timeout=timeout_s,
            gc_after_trial=True,
        )
    finally:
        print("Melhor trial:", study.best_trial.number, "val_loss:", study.best_value)
        print("Hparams:", study.best_trial.params)

        # salva artefatos do melhor trial para retomar treino
        best_dir = os.path.join(cfg.get("output_dir", "checkpoints"), "best")
        os.makedirs(best_dir, exist_ok=True)
        # copia o checkpoint best do trial:
        trial_dir = os.path.join(
            cfg.get("output_dir", "checkpoints"),
            "trials",
            phase,
            f"trial_{study.best_trial.number:04d}",
        )
        best_ckpt = os.path.join(trial_dir, "best.ckpt")
        if os.path.exists(best_ckpt):
            import shutil

            shutil.copy2(best_ckpt, os.path.join(best_dir, f"{phase}_best_trial.ckpt"))
        # salva hparams
        with open(
            os.path.join(best_dir, f"{phase}_best_hparams.yaml"), "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(
                study.best_trial.params, f, sort_keys=False, allow_unicode=True
            )

        print(f"[OK] artefatos salvos em: {best_dir}")


if __name__ == "__main__":
    main()
