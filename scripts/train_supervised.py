import os
import sys

# Adiciona a raiz do projeto no path para encontrar o pacote `model`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import ChessSupervisedDataset
from model.network import ChessPolicyNetwork


def load_config(path="utils/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def topk_accuracy(preds, targets, k=3):
    """
    Computa a acurácia top-k: fração de exemplos onde o target
    está entre as top-k previsões.
    preds: [batch, num_moves]
    targets: [batch, num_moves] (distribuição)
    """
    true_idx = targets.argmax(dim=1)
    topk = preds.topk(k, dim=1).indices
    match = (topk == true_idx.unsqueeze(1)).any(dim=1)
    return match.float().mean().item()


def train():
    cfg = load_config()

    # batch_size, history_size e mode podem vir do root do config ou usar defaults
    batch_size = cfg.get("batch_size", 32)
    history_size = cfg.get("history_size", 4)
    mode = cfg.get("mode", "top3")

    # descobre a pasta de dados gerados
    output_data_dir = cfg["output_data_dir"]
    phases = cfg["phases"]
    if not phases:
        raise ValueError("Nenhuma phase definida em config.yaml em `phases`.")
    phase = next(iter(phases))  # pega a primeira phase (ex: 'v1_depth6')
    data_path = os.path.join(output_data_dir, phase, "stockfish_data.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset não encontrado em: {data_path}")

    # dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # carrega dados
    import pickle

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # monta Dataset e DataLoader
    dataset = ChessSupervisedDataset(
        data,
        use_heuristics=cfg.get("use_heuristics", True),
        mode=mode,
        history_size=history_size,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # instância da rede
    num_moves = dataset.num_moves
    net = ChessPolicyNetwork(
        num_moves=num_moves,
        history_size=history_size,
        heur_dim=8,  # now we have exactly 8 heuristic features
        use_heuristics=cfg.get("use_heuristics", True),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss()

    # loop de treino
    for epoch in range(1, cfg.get("epochs", 10) + 1):
        net.train()
        running_loss = 0.0
        running_top1 = 0.0
        running_top3 = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.get('epochs',10)}")
        for (
            half_moves_n_turn,
            board_tensor,
            heur_tensor,
            policy_target,
            legal_mask,
        ) in pbar:
            half_moves_n_turn_tensor = half_moves_n_turn.to(device)
            board_tensor = board_tensor.to(device)
            heur_tensor = heur_tensor.to(device)
            policy_target = policy_target.to(device)
            legal_mask = legal_mask.to(device)

            logits = net(board_tensor, heur_tensor, half_moves_n_turn_tensor)
            # -- action masking: invalida (–inf) as jogadas ilegais
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            target_idx = policy_target.argmax(dim=1)

            loss = criterion(masked_logits, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                probs = torch.softmax(masked_logits, dim=1)
                running_top1 += topk_accuracy(probs, policy_target, k=1)
                running_top3 += topk_accuracy(probs, policy_target, k=3)

            avg_loss = running_loss / len(loader)
            avg_top1 = running_top1 / len(loader)
            avg_top3 = running_top3 / len(loader)
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "top1": f"{avg_top1:.3f}",
                    "top3": f"{avg_top3:.3f}",
                }
            )

        # salvar checkpoint
        ckpt_dir = cfg.get("output_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            net.state_dict(), os.path.join(ckpt_dir, f"supervised_epoch{epoch}.pt")
        )

    print("Treino finalizado!")


if __name__ == "__main__":
    print("hello")
    train()
