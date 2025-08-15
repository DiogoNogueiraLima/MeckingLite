# scripts/evaluate_vs_stockfish.py
# -*- coding: utf-8 -*-

"""
Roda partidas entre Stockfish (profundidade fixa) e o seu modelo supervisionado,
sem salvar PGN; imprime apenas o placar final.

Compatível com:
  - model/dataset.py (board_to_tensor, get_move_index_map, evaluate_all)
  - model/network.py  (ChessPolicyNetwork)
  - checkpoints/supervised_epoch10.pt (state_dict)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ===================== CONFIGURAÇÃO GLOBAL ===================== #
# Ajuste apenas estas constantes:
STOCKFISH_PATH = Path(
    r"C:\Users\diogo\Repositorios\MeckingLite\engines\stockfish\stockfish\stockfish.exe"
)  # <- defina o seu .exe aqui
CHECKPOINT_PATH = Path(
    r"C:\Users\diogo\Repositorios\MeckingLite\checkpoints\supervised_v1_depth6_end_epoch4.ckpt"
)
GAMES = 1000
DEPTH = 1
DEVICE: Optional[str] = None  # "cuda" | "cpu" | None (auto)
# =============================================================== #

# Garante import do pacote raiz (…/MeckingLite)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import random
import yaml
import torch
import chess
import chess.engine
import numpy as np

# ---- imports do seu código ----
from model.dataset import board_to_tensor, get_move_index_map  # type: ignore
from model.heuristics.basic_heuristics import evaluate_all  # type: ignore
from model.network import ChessPolicyNetwork  # type: ignore


# --------------------- Utilidades --------------------- #
def load_config(path: Path) -> dict:
    if not path.exists():
        return {"history_size": 0, "use_heuristics": True}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_inputs(
    board: chess.Board,
    history_size: int,
    use_heuristics: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Tensores compatíveis com o treino:
      - board_tensor: [1, 12*(history_size+1), 8, 8]
      - heur_tensor:  [1, 8] (ou None)
      - half_moves_n_turn: [1, 2]
    Histórico é preenchido com zeros (padding).
    """
    board_np = board_to_tensor(board)  # [12,8,8]
    if history_size > 0:
        hist = np.zeros((12 * history_size, 8, 8), dtype=np.float32)
        board_np = np.concatenate([hist, board_np], axis=0)
    board_tensor = torch.from_numpy(board_np).unsqueeze(0)  # [1,C,8,8]

    heur = None
    if use_heuristics:
        mat, mob, ks, main_cc, ext_cc, wb, bb, total = evaluate_all(board)
        heur = torch.tensor(
            [[mat, mob, ks, main_cc, ext_cc, wb, bb, total]], dtype=torch.float32
        )

    turn_flag = float(board.turn)
    half_moves = (board.fullmove_number - 1) * 2 + int(board.turn == chess.BLACK)
    half_moves_n_turn = torch.tensor(
        [[half_moves / 40.0, turn_flag]], dtype=torch.float32
    )

    return board_tensor, heur, half_moves_n_turn


def legal_mask_for_board(
    board: chess.Board, move_index_map: Dict[str, int], num_moves: int
) -> torch.Tensor:
    """
    Máscara de jogadas legais usando a mesma convenção do dataset:
    promoções normalizadas para 4 chars (ex.: 'e7e8q' -> 'e7e8').
    """
    mask = torch.zeros(num_moves, dtype=torch.bool)
    for mv in board.legal_moves:
        uci = mv.uci()
        if len(uci) == 5 and uci[-1].lower() == "q":
            uci = uci[:-1]
        idx = move_index_map.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def idx_to_uci4(idx: int, inv_map: Dict[int, str]) -> Optional[str]:
    return inv_map.get(idx)


def uci4_to_legal_move(board: chess.Board, uci4: str) -> Optional[chess.Move]:
    """
    Converte 'e7e8' (4 chars) para um lance legal real.
    Se for promoção, prioriza dama ('q').
    """
    candidates = [mv for mv in board.legal_moves if mv.uci().startswith(uci4)]
    if not candidates:
        return None
    for mv in candidates:
        if mv.uci().endswith("q"):
            return mv
    return candidates[0]


# --- ADD (próximo ao ModelAgent) ---
class RandomAgent:
    @staticmethod
    def select(board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None


# --------------------- Agente do Modelo --------------------- #
class ModelAgent:
    def __init__(
        self, checkpoint: Path, device: Optional[str], cfg: dict, num_moves: int
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.history_size = int(cfg.get("history_size", 0))
        self.use_heuristics = bool(cfg.get("use_heuristics", True))

        self.net = (
            ChessPolicyNetwork(
                num_moves=num_moves,
                history_size=self.history_size,
                heur_dim=8,
                use_heuristics=self.use_heuristics,
            )
            .to(self.device)
            .eval()
        )

        state = torch.load(checkpoint, map_location=self.device)
        self.net.load_state_dict(state, strict=False)

    @torch.no_grad()
    def select(
        self,
        board: chess.Board,
        legal_mask: torch.Tensor,
        move_index_map: Dict[str, int],
        inv_index_map: Dict[int, str],
    ) -> Optional[chess.Move]:
        bt, ht, hmt = build_inputs(board, self.history_size, self.use_heuristics)
        bt = bt.to(self.device)
        hmt = hmt.to(self.device)
        if ht is not None:
            ht = ht.to(self.device)

        logits = self.net(bt, ht, hmt)  # [1, num_moves]
        masked = logits.masked_fill(~legal_mask.to(self.device), float("-inf"))
        idx = int(torch.softmax(masked, dim=1).argmax(dim=1).item())

        uci4 = idx_to_uci4(idx, inv_index_map)
        if uci4 is None:
            return None
        return uci4_to_legal_move(board, uci4)


# --------------------- Partida & Avaliação --------------------- #
# --- REPLACE a assinatura de play_game para usar random_agent em vez de Stockfish ---
def play_game(
    random_agent: RandomAgent,
    model_agent: ModelAgent,
    move_index_map: Dict[str, int],
    inv_index_map: Dict[int, str],
    model_is_white: bool,
) -> str:
    board = chess.Board()
    num_moves = len(inv_index_map)

    while not board.is_game_over():
        if (board.turn and model_is_white) or (
            (not board.turn) and (not model_is_white)
        ):
            legal_mask = legal_mask_for_board(board, move_index_map, num_moves)
            mv = model_agent.select(board, legal_mask, move_index_map, inv_index_map)
            if mv is None:
                return "0-1" if model_is_white else "1-0"
            board.push(mv)
        else:
            mv = random_agent.select(board)
            if mv is None:
                return "1-0" if model_is_white else "0-1"
            board.push(mv)

    out = board.outcome()
    return out.result() if out is not None else "*"


# --- REPLACE dentro de run_matches: NÃO abrir Stockfish; criar RandomAgent e chamar play_game ---
def run_matches(
    stockfish_path: Path,
    depth: int,
    games: int,
    checkpoint: Path,
    device: Optional[str],
) -> None:
    move_index_map, num_moves = get_move_index_map()
    inv_index_map = {v: k for k, v in move_index_map.items()}

    cfg = load_config(ROOT / "utils" / "config.yaml")
    agent = ModelAgent(
        checkpoint=checkpoint, device=device, cfg=cfg, num_moves=num_moves
    )
    rand = RandomAgent()

    w = d = l = 0
    for g in range(games):
        model_white = g % 2 == 0
        res = play_game(rand, agent, move_index_map, inv_index_map, model_white)
        if res == "1-0":
            if model_white:
                w += 1
            else:
                l += 1
        elif res == "0-1":
            if model_white:
                l += 1
            else:
                w += 1
        else:
            d += 1
        if (g + 1) % 50 == 0:
            print(f"[{g+1}/{games}] Parcial — W:{w} D:{d} L:{l}")

    points = w + 0.5 * d
    print("\n=== Resultado Final ===")
    print(f"Jogos: {games} | Adversário: Random")
    print(
        f"Modelo — Vitórias: {w}  Empates: {d}  Derrotas: {l}  | Pontos: {points:.1f}/{games}"
    )


# --------------------- Main --------------------- #
if __name__ == "__main__":
    run_matches(
        stockfish_path=STOCKFISH_PATH,
        depth=DEPTH,
        games=GAMES,
        checkpoint=CHECKPOINT_PATH,
        device=DEVICE,
    )
