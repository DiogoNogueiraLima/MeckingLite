import os
import sys
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# permite importar módulos do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.composite import CompositeDataset
from model.network import ChessPolicyNetwork

torch.backends.cudnn.benchmark = True  # acelera convs com shapes estáveis


def load_config():
    config_path = Path(__file__).parent.parent / "utils\\config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:  # <-- forçando UTF-8
        return yaml.safe_load(f)


def topk_accuracy(preds, targets, k=3):
    """preds: probas [B, A]; targets: distribuição [B, A] (ou one-hot)."""
    true_idx = targets.argmax(dim=1)
    topk = preds.topk(k, dim=1).indices
    match = (topk == true_idx.unsqueeze(1)).any(dim=1)
    return match.float().mean().item()


def latest_checkpoint(ckpt_dir, phase):
    patt = os.path.join(ckpt_dir, f"supervised_{phase}_*.ckpt")
    files = glob.glob(patt)
    if not files:
        return None
    # mais novo por mtime
    return max(files, key=os.path.getmtime)


def save_ckpt(path, model, optimizer, scaler, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "meta": meta,
        },
        path,
    )


def load_ckpt(path, model, optimizer, scaler):
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
            return False  # não para
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


def train():
    cfg = load_config()

    # --- hiperparâmetros básicos ---
    batch_size = cfg.get("batch_size", 32)
    history_size = cfg.get("history_size", 4)
    mode = cfg.get("mode", "top1")  # "top1" ou "top3"
    use_soft = mode == "top3"
    use_amp = bool(cfg.get("use_amp", True))
    grad_clip = float(cfg.get("grad_clip", 0.0))
    ckpt_dir = cfg.get("output_dir", "checkpoints")

    # early stopping (só no stage "end")
    es_cfg = cfg.get("early_stop", {})
    es_pat = int(es_cfg.get("patience", 3))
    es_delta = float(es_cfg.get("min_delta", 0.001))
    es_max_ep = int(es_cfg.get("max_epochs", 1000))  # teto de segurança

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- fase + currículo ---
    output_data_dir = cfg["output_data_dir"]
    phase = cfg.get("train_phase", "v1_depth6")
    phase_dir = os.path.join(output_data_dir, phase)
    if not os.path.isdir(phase_dir):
        raise FileNotFoundError(f"Diretório da fase não encontrado: {phase_dir}")

    phase_key = f"phase_{phase}"
    schedule = cfg["curriculum"][phase_key]["sample_weights"]

    # --- datasets ---
    train_ds = CompositeDataset(
        phase_dir,
        history_size=history_size,
        use_heuristics=cfg.get("use_heuristics", True),
        mode=mode,
        weights=schedule["start"],  # começa em start
    )
    val_ds = CompositeDataset(
        phase_dir,
        history_size=history_size,
        use_heuristics=cfg.get("use_heuristics", True),
        mode=mode,
        weights=schedule["start"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- modelo ---
    # pega num_moves de qualquer bucket populado
    for tag in ("easy", "medium", "hard"):
        if len(train_ds.ds[tag]) > 0:
            num_moves = train_ds.ds[tag].num_moves
            break
    else:
        raise RuntimeError("Todos os buckets estão vazios.")

    net = ChessPolicyNetwork(
        num_moves=num_moves,
        history_size=history_size,
        heur_dim=8,
        use_heuristics=cfg.get("use_heuristics", True),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 1e-3))
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- estágios: start/middle fixos; end com early stopping ---
    stages = [("start", 2), ("middle", 2), ("end", None)]
    stage_names = [s for s, _ in stages]

    # --- resume ---
    resume_path = cfg.get("resume", "").strip()
    if not resume_path:
        auto = latest_checkpoint(ckpt_dir, phase)
        resume_path = auto or ""

    start_stage_idx = 0
    start_epoch_in_stage = 1
    global_step = 0

    if resume_path and os.path.isfile(resume_path):
        meta = load_ckpt(resume_path, net, optimizer, scaler)
        if meta:
            if meta.get("phase") == phase:
                if meta.get("stage_name") in stage_names:
                    start_stage_idx = stage_names.index(meta["stage_name"])
                    start_epoch_in_stage = meta.get("epoch", 1) + 1
                global_step = meta.get("global_step", 0)
            print(
                f"[RESUME] retomando de {resume_path} | stage={meta.get('stage_name')} epoch={meta.get('epoch')}"
            )
        else:
            print(f"[RESUME] carregado {resume_path} (sem meta detalhada)")

    # --- treino ---
    try:
        for sidx in range(start_stage_idx, len(stages)):
            stage_name, stage_epochs = stages[sidx]
            print(f"\n==== Stage: {stage_name} ({phase}) ====")
            train_ds.set_stage(schedule, stage_name)
            val_ds.set_stage(schedule, stage_name)

            # setup do iterador de épocas
            if stage_name == "end":
                stopper = EarlyStopper(patience=es_pat, min_delta=es_delta)
                epoch_iter = range(
                    start_epoch_in_stage if sidx == start_stage_idx else 1,
                    es_max_ep + 1,
                )
            else:
                epoch_iter = range(
                    start_epoch_in_stage if sidx == start_stage_idx else 1,
                    (stage_epochs or 0) + 1,
                )

            for epoch in epoch_iter:
                # --- treino ---
                net.train()
                running_loss = running_top1 = running_top3 = 0.0
                pbar = tqdm(
                    train_loader,
                    desc=f"{stage_name} | Epoch {epoch}/{stage_epochs if stage_epochs else '∞'}",
                )

                for (
                    half_moves_n_turn,
                    board_tensor,
                    heur_tensor,
                    policy_target,
                    legal_mask,
                ) in pbar:

                    half_moves_n_turn = half_moves_n_turn.to(device)
                    board_tensor = board_tensor.to(device)
                    heur_tensor = heur_tensor.to(device)
                    policy_target = policy_target.to(device)
                    legal_mask = legal_mask.to(device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = net(board_tensor, heur_tensor, half_moves_n_turn)
                        masked_logits = logits.masked_fill(~legal_mask, float("-inf"))

                        if use_soft:
                            log_probs = torch.log_softmax(masked_logits, dim=1)
                            loss = criterion_kl(log_probs, policy_target)
                        else:
                            target_idx = policy_target.argmax(dim=1)
                            loss = criterion_ce(masked_logits, target_idx)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    with torch.no_grad():
                        probs = torch.softmax(masked_logits, dim=1)
                        running_top1 += topk_accuracy(probs, policy_target, k=1)
                        running_top3 += topk_accuracy(probs, policy_target, k=3)

                    global_step += 1
                    avg_loss = running_loss / max(1, len(train_loader))
                    avg_top1 = running_top1 / max(1, len(train_loader))
                    avg_top3 = running_top3 / max(1, len(train_loader))
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "top1": f"{avg_top1:.3f}",
                            "top3": f"{avg_top3:.3f}",
                        }
                    )

                # --- validação ---
                net.eval()
                val_loss = 0.0
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    for (
                        half_moves_n_turn,
                        board_tensor,
                        heur_tensor,
                        policy_target,
                        legal_mask,
                    ) in val_loader:

                        half_moves_n_turn = half_moves_n_turn.to(device)
                        board_tensor = board_tensor.to(device)
                        heur_tensor = heur_tensor.to(device)
                        policy_target = policy_target.to(device)
                        legal_mask = legal_mask.to(device)

                        logits = net(board_tensor, heur_tensor, half_moves_n_turn)
                        masked_logits = logits.masked_fill(~legal_mask, float("-inf"))

                        if use_soft:
                            log_probs = torch.log_softmax(masked_logits, dim=1)
                            loss = criterion_kl(log_probs, policy_target)
                        else:
                            target_idx = policy_target.argmax(dim=1)
                            loss = criterion_ce(masked_logits, target_idx)

                        val_loss += loss.item()

                val_loss /= max(1, len(val_loader))
                print(f"Val loss ({stage_name}): {val_loss:.4f}")

                # --- checkpoint desta época ---
                ckpt_path = os.path.join(
                    ckpt_dir, f"supervised_{phase}_{stage_name}_epoch{epoch}.ckpt"
                )
                meta = {
                    "phase": phase,
                    "stage_name": stage_name,
                    "stage_idx": sidx,
                    "epoch": epoch,
                    "global_step": global_step,
                    "num_moves": num_moves,
                    "mode": mode,
                }
                save_ckpt(ckpt_path, net, optimizer, scaler, meta)

                # early stop apenas no "end"
                if stage_name == "end" and stopper.step(val_loss):
                    print(
                        f"[EARLY STOP] Parou no epoch {epoch} (best val_loss={stopper.best:.4f})"
                    )
                    break

            # próximo estágio começa do epoch 1
            start_epoch_in_stage = 1

    except KeyboardInterrupt:
        # salva checkpoint mesmo se interromper cedo
        safe_stage_name = locals().get("stage_name", "unknown")
        safe_sidx = locals().get("sidx", 0)
        safe_epoch = locals().get("epoch", 0)

        ckpt_path = os.path.join(ckpt_dir, f"supervised_{phase}_INTERRUPTED.ckpt")
        meta = {
            "phase": phase,
            "stage_name": safe_stage_name,
            "stage_idx": safe_sidx,
            "epoch": max(1, safe_epoch),
            "global_step": global_step,
            "num_moves": num_moves,
            "mode": mode,
        }
        save_ckpt(ckpt_path, net, optimizer, scaler, meta)
        print(f"\n[INTERRUPT] checkpoint salvo em: {ckpt_path}")

    print("Treino finalizado!")


if __name__ == "__main__":
    train()
