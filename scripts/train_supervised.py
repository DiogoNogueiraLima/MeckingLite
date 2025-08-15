# scripts/train_supervised.py
import os, sys, glob, yaml, time, csv, math, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

# --- matplotlib para salvar PNGs (sem abrir janela) ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- AMP API moderna ---
from torch import amp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.composite import CompositeDataset
from model.network import ChessPolicyNetwork  # sua rede

torch.backends.cudnn.benchmark = True


# ---------- utils ----------
def load_config():
    config_path = Path(__file__).parent.parent / "utils\\config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def topk_accuracy(probs, targets, k=3):
    true_idx = targets.argmax(dim=1)
    topk = probs.topk(k, dim=1).indices
    match = (topk == true_idx.unsqueeze(1)).any(dim=1)
    return match.float().mean().item()


def mrr_at_k(probs, targets, k=10):
    true_idx = targets.argmax(dim=1)
    topk = probs.topk(k, dim=1).indices
    ranks = torch.full((probs.size(0),), fill_value=0.0, device=probs.device)
    for r in range(k):
        ranks = torch.where((topk[:, r] == true_idx), 1.0 / (r + 1), ranks)
    return ranks.mean().item()


def entropy(probs, eps=1e-9):
    p = torch.clamp(probs, eps, 1.0)
    ent = -(p * p.log()).sum(dim=1)
    return ent.mean().item()


def expected_calibration_error(probs, targets, n_bins=15):
    # ECE simples: |acc - conf| ponderado por massa de cada bin
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        true = targets.argmax(dim=1)
        ece = 0.0
        bins = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                acc = (pred[mask] == true[mask]).float().mean()
                avg_conf = conf[mask].mean()
                ece += (mask.float().mean() * (acc - avg_conf).abs()).item()
        return float(ece)


def latest_checkpoint(ckpt_dir, phase):
    patt = os.path.join(ckpt_dir, f"supervised_{phase}_*.ckpt")
    files = glob.glob(patt)
    return max(files, key=os.path.getmtime) if files else None


def save_ckpt(path, model, optimizer, scaler, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "meta": meta,
        },
        path,
    )


def load_ckpt(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("meta", {})


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


# ---------- helpers de plots (PNGs) ----------
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
    hist: {
      'epoch': [1,2,...],
      'train': {'loss':[], 'top1':[], 'top3':[], 'mrr10':[], 'ece':[], 'entropy':[]},
      'val':   {'loss':[], 'top1':[], 'top3':[], 'mrr10':[], 'ece':[], 'entropy':[]},
    }
    """
    _ensure_dir(out_dir)
    epochs = hist["epoch"]
    if not epochs:
        return
    tr = hist["train"]
    va = hist["val"]

    # Visão geral 2x3
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

    # Métricas individuais
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


# ---------- fábricas para reuso ----------
def make_datasets_and_loaders(cfg, weights_stage, batch_size=None, pin=True, workers=2):
    output_data_dir = cfg["output_data_dir"]
    phase = cfg.get("train_phase", "v1_depth6")
    phase_dir = os.path.join(output_data_dir, phase)
    if not os.path.isdir(phase_dir):
        raise FileNotFoundError(f"Diretório da fase não encontrado: {phase_dir}")

    train_ds = CompositeDataset(
        phase_dir,
        history_size=cfg.get("history_size", 0),
        use_heuristics=cfg.get("use_heuristics", True),
        mode=cfg.get("mode", "top3"),
        weights=weights_stage,
    )
    val_ds = CompositeDataset(
        phase_dir,
        history_size=cfg.get("history_size", 0),
        use_heuristics=cfg.get("use_heuristics", True),
        mode=cfg.get("mode", "top3"),
        weights=weights_stage,
    )

    for tag in ("easy", "medium", "hard"):
        if len(train_ds.ds[tag]) > 0:
            num_moves = train_ds.ds[tag].num_moves
            break
    else:
        raise RuntimeError("Todos os buckets estão vazios.")

    bs = batch_size or cfg.get("batch_size", 256)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=True,
    )
    return (train_ds, val_ds, train_loader, val_loader, num_moves)


def make_model(cfg, num_moves):
    net = ChessPolicyNetwork(
        num_moves=num_moves,
        history_size=cfg.get("history_size", 0),
        heur_dim=8,
        use_heuristics=cfg.get("use_heuristics", True),
        half_moves_n_turn_dim=4,  # [halfmove_norm, turn_flag, can_castle_K, can_castle_Q]
        dropout=cfg.get("dropout", 0.1),
    )
    return net


# ---------- perdas ----------
def _build_topk_target(policy_target, legal_mask, k, temperature=1.0):
    """Retorna distribuição alvo restrita ao top-k (entre os LEGAIS) e renormalizada."""
    masked_target = policy_target * legal_mask.float()
    # se não houver massa, cai pra uniforme nos legais
    sums = masked_target.sum(dim=1, keepdim=True)
    zero_mass = sums.squeeze(1) <= 0
    if zero_mass.any():
        uni = legal_mask[zero_mass].float()
        uni = uni / uni.sum(dim=1, keepdim=True)
        masked_target[zero_mass] = uni
        sums = masked_target.sum(dim=1, keepdim=True)

    # top-k nos legais
    k = max(1, int(k))
    topk_vals, topk_idx = masked_target.topk(k, dim=1)
    tk = torch.zeros_like(policy_target)
    tk.scatter_(1, topk_idx, topk_vals)

    # “afiar” com temperatura opcional
    tk = tk.clamp_min(1e-12)
    if temperature != 1.0:
        tk = tk.pow(1.0 / float(temperature))

    tk = tk / tk.sum(dim=1, keepdim=True)
    return tk


def _pairwise_rank_loss(
    masked_logits, policy_target, legal_mask, negatives=4, margin=1.0
):
    """
    Ranking par-a-par: para cada amostra, compara best vs m negativos legais.
    Usa softplus(margin - (pos - neg)) (estável).
    """
    B, A = masked_logits.shape
    # índice do "melhor" no alvo
    best_idx = policy_target.argmax(dim=1)

    # coleta índices legais por amostra
    # (evita casos com nenhum legal; fallback no próprio best_idx)
    neg_list = []
    for b in range(B):
        legal_ids = legal_mask[b].nonzero(as_tuple=False).squeeze(1)
        # remove o best
        legal_ids = legal_ids[legal_ids != best_idx[b]]
        if legal_ids.numel() == 0:
            # fallback: repete best (sem penalizar)
            neg_list.append(best_idx[b].unsqueeze(0).repeat(negatives))
        else:
            m = min(negatives, legal_ids.numel())
            choice = legal_ids[torch.randint(low=0, high=legal_ids.numel(), size=(m,))]
            # completa se legal_ids < negatives
            if m < negatives:
                pad = choice[torch.randint(low=0, high=m, size=(negatives - m,))]
                choice = torch.cat([choice, pad], dim=0)
            neg_list.append(choice)
    neg_idx = torch.stack(neg_list, dim=0)  # [B, negatives]

    pos = masked_logits.gather(1, best_idx.unsqueeze(1))  # [B,1]
    neg = masked_logits.gather(1, neg_idx)  # [B,m]

    # softplus(margin - (pos - neg)) é estável
    rank_loss = torch.nn.functional.softplus(margin - (pos - neg)).mean()
    return rank_loss


# ---------- passo de treino/val ----------
def run_one_epoch(
    net,
    loader,
    device,
    use_amp,
    loss_mode,  # "pairwise" | "topk_kl" | "kl" | "ce" | "blend" (se quiser)
    loss_params,  # dict com parâmetros da loss
    criterion_ce,
    criterion_kl,
    optimizer=None,
    scaler=None,
    grad_clip=0.0,
):
    is_train = optimizer is not None
    net.train() if is_train else net.eval()

    loss_sum = 0.0
    n_batches = 0
    top1_sum = 0.0
    top3_sum = 0.0
    mrr_sum = 0.0
    ent_sum = 0.0
    ece_sum = 0.0

    t0 = time.time()
    for (
        half_moves_n_turn,
        board_tensor,
        heur_tensor,
        policy_target,
        legal_mask,
    ) in loader:
        # --- envio p/ device e formatação channel-last ---
        board_tensor = board_tensor.to(device).to(memory_format=torch.channels_last)
        heur_tensor = heur_tensor.to(device)
        policy_target = policy_target.to(device)
        legal_mask = legal_mask.to(device)
        half_moves_n_turn = half_moves_n_turn.to(device)

        # compat: garantir 4 features (halfmove, turn, can_K, can_Q)
        if half_moves_n_turn.dim() == 2 and half_moves_n_turn.size(1) < 4:
            pad = torch.zeros(
                half_moves_n_turn.size(0),
                4 - half_moves_n_turn.size(1),
                device=half_moves_n_turn.device,
                dtype=half_moves_n_turn.dtype,
            )
            half_moves_n_turn = torch.cat([half_moves_n_turn, pad], dim=1)
        elif half_moves_n_turn.dim() == 2 and half_moves_n_turn.size(1) > 4:
            half_moves_n_turn = half_moves_n_turn[:, :4]

        # ---------------- SANITIZAÇÕES + MÁSCARA FINITA ----------------
        # A) Sanitiza alvo bruto (remove NaN/Inf/negativos residuais)
        policy_target = policy_target.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        policy_target = torch.clamp(policy_target, min=0.0)

        # B) Garante pelo menos 1 lance legal por amostra
        no_legal = ~legal_mask.any(dim=1)
        if no_legal.any():
            fallback_idx = policy_target.argmax(dim=1)
            legal_mask = legal_mask.clone()
            legal_mask[no_legal] = False
            legal_mask.scatter_(1, fallback_idx.unsqueeze(1), True)

        # C) Se usaremos "kl" (completo), projetamos o alvo em legais + normaliza
        if loss_mode == "kl":
            pt = policy_target * legal_mask.float()
            s = pt.sum(dim=1, keepdim=True)
            zero_mass = s.squeeze(1) <= 0
            if zero_mass.any():
                uni = legal_mask[zero_mass].float()
                uni = uni / uni.sum(dim=1, keepdim=True)
                pt[zero_mass] = uni
                s = pt.sum(dim=1, keepdim=True)
            policy_target_eff = pt / s
        else:
            policy_target_eff = policy_target  # outros modos tratam seu próprio alvo

        # ---------------- FORWARD ----------------
        with amp.autocast(device_type="cuda", enabled=use_amp):
            logits = net(board_tensor, heur_tensor, half_moves_n_turn)

        # Aplica máscara com valor finito e faz perda em FP32 para estabilidade
        masked_logits = logits.float().masked_fill(~legal_mask, -1e9)
        masked_logits = masked_logits.clamp(min=-80.0, max=80.0)

        # ---------------- LOSS por modo ----------------
        if loss_mode == "pairwise":
            m = int(loss_params.get("negatives", 4))
            margin = float(loss_params.get("margin", 1.0))
            loss = _pairwise_rank_loss(
                masked_logits, policy_target_eff, legal_mask, negatives=m, margin=margin
            )

        elif loss_mode == "topk_kl":
            k = int(loss_params.get("topk", 3))
            T = float(loss_params.get("temperature", 1.0))
            tk = _build_topk_target(policy_target_eff, legal_mask, k=k, temperature=T)
            log_probs = torch.log_softmax(masked_logits, dim=1)
            loss = criterion_kl(log_probs, tk)

        elif loss_mode == "kl":
            log_probs = torch.log_softmax(masked_logits, dim=1)
            loss = criterion_kl(log_probs, policy_target_eff)

        elif loss_mode == "ce":
            target_idx = policy_target.argmax(dim=1)
            is_illegal = ~legal_mask.gather(1, target_idx.unsqueeze(1)).squeeze(1)
            if is_illegal.any():
                fallback_idx = legal_mask.float().argmax(dim=1)
                target_idx = torch.where(is_illegal, fallback_idx, target_idx)
            loss = criterion_ce(masked_logits, target_idx)

        elif loss_mode == "blend":
            # opcional: mistura CE(top-1) com KL(soft)
            alpha = float(loss_params.get("alpha", 0.6))
            log_probs = torch.log_softmax(masked_logits, dim=1)
            loss_ce = criterion_ce(masked_logits, policy_target.argmax(dim=1))
            loss_kl = criterion_kl(log_probs, policy_target_eff)
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kl

        else:
            raise ValueError(f"loss_mode desconhecido: {loss_mode}")

        # Proteção final: se NaN/Inf, pula o batch com log
        if not torch.isfinite(loss):
            n_no_legal = int(no_legal.sum().item())
            if loss_mode in ("kl", "topk_kl", "blend"):
                n_zero_mass = int(((policy_target_eff.sum(dim=1)) <= 0).sum().item())
            else:
                n_zero_mass = 0
            n_bad_logit = int((~torch.isfinite(masked_logits)).any(dim=1).sum().item())
            print(
                f"[WARN] loss NaN/Inf: skip batch | no_legal={n_no_legal} zero_mass={n_zero_mass} bad_logit_rows={n_bad_logit} mode={loss_mode}"
            )
            continue

        # ---------------- BACKWARD ----------------
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # ---------------- MÉTRICAS ----------------
        with torch.no_grad():
            probs = torch.softmax(masked_logits, dim=1)  # FP32 estável
            loss_sum += float(loss.item())
            n_batches += 1
            # Para métricas, use o alvo efetivo quando fizer sentido
            tgt_for_metrics = (
                policy_target_eff
                if loss_mode in ("kl", "topk_kl", "blend")
                else policy_target
            )
            top1_sum += topk_accuracy(probs, tgt_for_metrics, k=1)
            top3_sum += topk_accuracy(probs, tgt_for_metrics, k=3)
            mrr_sum += mrr_at_k(probs, tgt_for_metrics, k=10)
            ent_sum += entropy(probs)
            ece_sum += expected_calibration_error(probs, tgt_for_metrics, n_bins=15)

    secs = time.time() - t0
    metrics = {
        "loss": loss_sum / max(1, n_batches),
        "top1": top1_sum / max(1, n_batches),
        "top3": top3_sum / max(1, n_batches),
        "mrr10": mrr_sum / max(1, n_batches),
        "entropy": ent_sum / max(1, n_batches),
        "ece": ece_sum / max(1, n_batches),
        "sec": secs,
        "it_per_sec": n_batches / secs if secs > 0 else 0.0,
    }
    return metrics


# ---------- treino final ----------
def train():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("use_amp", True))
    grad_clip = float(cfg.get("grad_clip", 0.0))
    ckpt_dir = cfg.get("output_dir", "checkpoints")

    phase = cfg.get("train_phase", "v1_depth6")
    phase_key = f"phase_{phase}"
    schedule = cfg["curriculum"][phase_key]["sample_weights"]

    batch_size = cfg.get("batch_size", 256)
    (train_ds, val_ds, train_loader, val_loader, num_moves) = make_datasets_and_loaders(
        cfg, schedule["start"], batch_size, workers=2
    )

    net = make_model(cfg, num_moves).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    # Stages do currículo (amostragem easy/medium/hard)
    stages = [("start", 2), ("middle", 2), ("end", None)]

    # Loss schedule por stage (pode sobrescrever no YAML em logging.loss_schedule)
    # default: start=pairwise, middle=topk_kl, end=kl
    loss_schedule = cfg.get(
        "loss_schedule",
        {
            "start": {
                "mode": "pairwise",
                "epochs": 2,
                "params": {"negatives": 4, "margin": 1.0},
            },
            "middle": {
                "mode": "topk_kl",
                "epochs": 3,
                "params": {"topk": 3, "temperature": 0.8},
            },
            "end": {"mode": "kl", "epochs": None, "params": {}},
        },
    )

    # Early stop
    es_cfg = cfg.get("early_stop", {})
    es_pat = int(es_cfg.get("patience", 3))
    es_delta = float(es_cfg.get("min_delta", 0.001))
    es_max_ep = int(es_cfg.get("max_epochs", 1000))
    epochs_end = int(cfg.get("epochs", 10))  # teto p/ "end" se quiser fixar

    # Logging dirs
    log_cfg = cfg.get("logging", {})
    tb_dir_root = log_cfg.get("tensorboard_dir", "runs")
    csv_dir = log_cfg.get("csv_dir", "logs")
    os.makedirs(csv_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(tb_dir_root, f"{phase}_final"))

    # Histórico para PNGs
    run_name = f"{phase}_final"
    png_dir = os.path.join(csv_dir, "png", run_name)
    _ensure_dir(png_dir)
    hist = {
        "epoch": [],
        "train": {k: [] for k in ["loss", "top1", "top3", "mrr10", "ece", "entropy"]},
        "val": {k: [] for k in ["loss", "top1", "top3", "mrr10", "ece", "entropy"]},
    }

    # Resume (automático do último, se config.resume vazio)
    resume_path = cfg.get("resume", "").strip()
    if not resume_path:
        auto = latest_checkpoint(ckpt_dir, phase)
        resume_path = auto or ""
    if resume_path and os.path.isfile(resume_path):
        meta = load_ckpt(resume_path, net, optimizer, scaler)
        print(f"[RESUME] de {resume_path} | meta={meta}")

    try:
        global_step = 0
        global_epoch = 0  # contador contínuo p/ gráficos

        for sidx, (stage_name, _stage_epochs_unused) in enumerate(stages):
            print(f"\n==== Stage: {stage_name} ({phase}) ====")
            train_ds.set_stage(schedule, stage_name)
            val_ds.set_stage(schedule, stage_name)

            # Escolhe perda e nº de épocas a partir do schedule
            stage_cfg = loss_schedule.get(stage_name, {})
            loss_mode = stage_cfg.get("mode", "kl")
            stage_epochs = stage_cfg.get("epochs", None)  # None = usa early stop no end
            loss_params = stage_cfg.get("params", {})

            stopper = None
            if stage_name == "end":
                # se não vier epochs no schedule, use early stop com teto 'epochs_end'
                if stage_epochs is None:
                    stage_epochs = epochs_end
                stopper = EarlyStopper(patience=es_pat, min_delta=es_delta)

            # CSV por run final
            csv_path = os.path.join(csv_dir, f"{phase}_final.csv")

            # Loop de épocas
            max_iter = int(stage_epochs) if stage_epochs else es_max_ep
            for epoch in range(1, max_iter + 1):
                tr = run_one_epoch(
                    net,
                    train_loader,
                    device,
                    use_amp,
                    loss_mode,
                    loss_params,
                    criterion_ce,
                    criterion_kl,
                    optimizer=optimizer,
                    scaler=scaler,
                    grad_clip=grad_clip,
                )
                va = run_one_epoch(
                    net,
                    val_loader,
                    device,
                    use_amp,
                    loss_mode,
                    loss_params,
                    criterion_ce,
                    criterion_kl,
                    optimizer=None,
                    scaler=None,
                )

                # TensorBoard
                for k, v in tr.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                for k, v in va.items():
                    writer.add_scalar(f"val/{k}", v, global_step)

                # CSV
                new_file = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow(
                            ["stage", "loss_mode", "epoch", "step"]
                            + [f"train_{k}" for k in tr.keys()]
                            + [f"val_{k}" for k in va.keys()]
                        )
                    w.writerow(
                        [stage_name, loss_mode, epoch, global_step]
                        + list(tr.values())
                        + list(va.values())
                    )

                # checkpoint por época
                meta = {
                    "phase": phase,
                    "stage_name": stage_name,
                    "epoch": epoch,
                    "global_step": global_step,
                    "num_moves": num_moves,
                    "loss_mode": loss_mode,
                    "loss_params": loss_params,
                }
                ckpt_path = os.path.join(
                    ckpt_dir, f"supervised_{phase}_{stage_name}_epoch{epoch}.ckpt"
                )
                save_ckpt(ckpt_path, net, optimizer, scaler, meta)

                # Print evolução
                print(
                    f"[{stage_name}/{loss_mode} ep{epoch}] "
                    f"train loss={tr['loss']:.4f} top1={tr['top1']:.3f} | "
                    f"val loss={va['loss']:.4f} top1={va['top1']:.3f} top3={va['top3']:.3f} "
                    f"mrr10={va['mrr10']:.3f} ece={va['ece']:.3f}"
                )

                # Atualiza histórico + PNGs
                global_epoch += 1
                hist["epoch"].append(global_epoch)
                for k in hist["train"].keys():
                    hist["train"][k].append(tr[k])
                    hist["val"][k].append(va[k])
                save_png_plots(run_name, hist, png_dir)

                global_step += 1

                # Early stop (apenas no "end" com stopper ativo)
                if stopper and stopper.step(va["loss"]):
                    print(f"[EARLY STOP] best val_loss={stopper.best:.4f}")
                    break

    except KeyboardInterrupt:
        ckpt_path = os.path.join(ckpt_dir, f"supervised_{phase}_INTERRUPTED.ckpt")
        save_ckpt(ckpt_path, net, optimizer, scaler, {"phase": phase})
        print(f"\n[INTERRUPT] checkpoint salvo em: {ckpt_path}")

    writer.close()
    print("Treino finalizado!")


if __name__ == "__main__":
    train()
