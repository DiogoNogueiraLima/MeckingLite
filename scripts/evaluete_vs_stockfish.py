# scripts/eval_topk_sf_pick.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys
import random
import time
import yaml
import torch
import chess
import chess.engine
import numpy as np

# ===================== CONFIG ===================== #
# >>> Ajuste estes caminhos/valores conforme necessário
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

STOCKFISH_PATH = Path(
    r"C:\Users\diogo\Repositorios\MeckingLite\engines\stockfish\stockfish\stockfish.exe"
)
CHECKPOINT_PATH = Path(
    r"C:\Users\diogo\Repositorios\MeckingLite\checkpoints\supervised_v1_depth6_middle_epoch3.ckpt"
)

# Avaliador (Stockfish) – rápido e previsível
SF_PICK_DEPTH = 12  # profundidade usada para escolher entre os Top-K do modelo
SF_PICK_THREADS = 1  # 1 thread (menor overhead)
SF_PICK_HASH_MB = 8  # hash pequeno basta para buscas curtas

# Geração de candidatos pelo modelo
TOPK = 8  # Top-K da rede para passar ao Stockfish (8–12 é ótimo)

# Oponentes e partidas
RUN_RANDOM = True
RUN_SF_D1 = True
RUN_SF_D2 = True
RUN_SF_D4 = True
GAMES_PER_OPP = 300

DEVICE: Optional[str] = None  # "cuda" | "cpu" | None (auto)
SEED = 42
# ================================================== #

# ---- imports do seu projeto ----
from model.dataset import board_to_tensor, get_move_index_map
from model.heuristics.basic_heuristics import evaluate_all as basic_eval
from model.network import ChessPolicyNetwork


# ---------------- Utilidades ---------------- #
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    cfg_path = path if path.exists() else (ROOT / "utils" / "config.yaml")
    if not cfg_path.exists():
        return {"history_size": 0, "use_heuristics": True}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_inputs(board: chess.Board, history_size: int, use_heuristics: bool):
    """
    Tensores compatíveis com o treino atual:
      - board_tensor: [1, 12*(H+1), 8, 8]
      - heur_tensor:  [1, 8] (basic_heuristics)
      - half_moves_n_turn: [1, 4] -> [halfmoves_norm, turn, canK, canQ]
    """
    # board
    bt = board_to_tensor(board)  # [12,8,8]
    if history_size > 0:
        hist = np.zeros((12 * history_size, 8, 8), dtype=np.float32)
        bt = np.concatenate([hist, bt], axis=0)
    board_tensor = torch.from_numpy(bt).unsqueeze(0)  # [1,C,8,8]

    # heurísticas (8 valores)
    h = basic_eval(board)
    heur_tensor = (
        torch.tensor([list(h)], dtype=torch.float32) if use_heuristics else None
    )

    # meta-features (4)
    turn_flag = float(board.turn)
    half_moves = (board.fullmove_number - 1) * 2 + int(board.turn == chess.BLACK)
    side = chess.WHITE if board.turn else chess.BLACK
    canK = float(board.has_kingside_castling_rights(side))
    canQ = float(board.has_queenside_castling_rights(side))
    half_moves_n_turn = torch.tensor(
        [[half_moves / 40.0, turn_flag, canK, canQ]], dtype=torch.float32
    )

    return board_tensor, heur_tensor, half_moves_n_turn


def legal_mask_for_board(
    board: chess.Board, move_index_map: Dict[str, int], num_moves: int
) -> torch.Tensor:
    mask = torch.zeros(num_moves, dtype=torch.bool)
    for mv in board.legal_moves:
        uci = mv.uci()
        # normaliza promo para 4 chars (e7e8q -> e7e8)
        if len(uci) == 5 and uci[-1].lower() == "q":
            uci = uci[:-1]
        idx = move_index_map.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def idx_to_uci4(idx: int, inv_map: Dict[int, str]) -> Optional[str]:
    return inv_map.get(idx)


def uci4_to_legal_move(board: chess.Board, uci4: str) -> Optional[chess.Move]:
    cands = [mv for mv in board.legal_moves if mv.uci().startswith(uci4)]
    if not cands:
        return None
    for mv in cands:
        if mv.uci().endswith("q"):
            return mv
    return cands[0]


# ---------------- Agentes ---------------- #
class RandomAgent:
    @staticmethod
    def select(board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None


class StockfishAgent:
    """Oponente que JOGA com Stockfish em profundidade fixa (rápido)."""

    def __init__(self, path: Path, depth: int):
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self.depth = int(depth)

    def select(self, board: chess.Board) -> Optional[chess.Move]:
        res = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return res.move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


class StockfishScorer:
    """
    Escolhe o melhor lance entre a lista de candidatos usando UMA chamada ao SF com:
      - searchmoves=<candidatos>
      - depth fixo (SF_PICK_DEPTH)
      - Threads=1, Hash=8 -> super rápido e determinístico
    """

    def __init__(self, path: Path):
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        try:
            self.engine.configure(
                {
                    "Threads": SF_PICK_THREADS,
                    "Hash": SF_PICK_HASH_MB,
                }
            )
        except Exception:
            # Alguns builds não aceitam configurar tudo — seguimos mesmo assim
            pass

    def pick(
        self, board: chess.Board, candidates: List[chess.Move]
    ) -> Optional[chess.Move]:
        if not candidates:
            return None
        limit = chess.engine.Limit(depth=SF_PICK_DEPTH)
        try:
            # UMA análise com searchmoves = candidatos -> SF escolhe o melhor deles
            info = self.engine.analyse(board, limit=limit, root_moves=candidates)
            # A API retorna um dict; mas o melhor lance é o que o SF jogaria:
            res = self.engine.play(board, limit=limit, root_moves=candidates)
            return res.move
        except Exception:
            # fallback: escolhe o primeiro
            return candidates[0]

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


class ModelPolicy:
    """
    Rede => Top-K lances plausíveis (com máscara de legais).
    """

    def __init__(self, checkpoint: Path, device: Optional[str], num_moves: int):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cfg = load_config(ROOT / "utils" / "config.yaml")
        self.history_size = int(cfg.get("history_size", 0))
        self.use_heuristics = bool(cfg.get("use_heuristics", True))

        self.net = (
            ChessPolicyNetwork(
                num_moves=num_moves,
                history_size=self.history_size,
                heur_dim=8,
                use_heuristics=self.use_heuristics,
                half_moves_n_turn_dim=4,  # [halfmoves_norm, turn, canK, canQ]
            )
            .to(self.device)
            .eval()
        )

        # suporta ckpt salvo com {"model": state_dict} ou direto state_dict
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        print(
            f"[DEBUG] load_state_dict -> missing={len(missing)} | unexpected={len(unexpected)}"
        )

        self.move_index_map, _ = get_move_index_map()
        self.inv_index_map = {v: k for k, v in self.move_index_map.items()}
        self.num_moves = len(self.inv_index_map)

    @torch.no_grad()
    def topk_moves(self, board: chess.Board, k: int) -> List[chess.Move]:
        bt, ht, hmt = build_inputs(board, self.history_size, self.use_heuristics)
        bt = bt.to(self.device).to(memory_format=torch.channels_last)
        hmt = hmt.to(self.device)
        if ht is not None:
            ht = ht.to(self.device)

        logits = self.net(bt, ht, hmt)  # [1, A]
        legal_mask = legal_mask_for_board(
            board, self.move_index_map, self.num_moves
        ).to(self.device)
        masked = logits.float().masked_fill(~legal_mask, -1e9)  # banir ilegais
        probs = torch.softmax(masked, dim=1).squeeze(0)

        legal_count = int(legal_mask.sum().item())
        if legal_count <= 0:
            return []
        kk = min(k, legal_count)

        top = torch.topk(probs, k=kk)
        out: List[chess.Move] = []
        used = set()
        for idx in top.indices.tolist():
            if idx in used:
                continue
            used.add(idx)
            u4 = idx_to_uci4(idx, self.inv_index_map)
            if not u4:
                continue
            mv = uci4_to_legal_move(board, u4)
            if mv is not None:
                out.append(mv)
        return out


class HybridAgent:
    """
    Agente híbrido:
      1) ModelPolicy sugere Top-K.
      2) StockfishScorer escolhe o melhor entre esses K com depth fixo (rápido).
    """

    def __init__(
        self, checkpoint: Path, device: Optional[str], sf_path: Path, num_moves: int
    ):
        self.policy = ModelPolicy(checkpoint, device, num_moves)
        self.scorer = StockfishScorer(sf_path)

    def select(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
        cand = self.policy.topk_moves(board, TOPK)
        return self.scorer.pick(board, cand)

    def close(self):
        self.scorer.close()


# ---------------- Partida / Avaliação ---------------- #
def play_game(agent_white, agent_black, max_fullmoves=200) -> str:
    board = chess.Board()
    fm0 = board.fullmove_number
    while not board.is_game_over():
        mv = agent_white.select(board) if board.turn else agent_black.select(board)
        if mv is None:
            break
        board.push(mv)
        if (board.fullmove_number - fm0) >= max_fullmoves:
            return "1/2-1/2"
    out = board.outcome()
    return out.result() if out else "*"


def run_matches_vs(opponent_name: str, opponent_factory, games: int):
    set_seed(SEED)
    move_index_map, num_moves = get_move_index_map()

    # nosso agente híbrido (modelo + SF para escolher entre K)
    agent = HybridAgent(CHECKPOINT_PATH, DEVICE, STOCKFISH_PATH, num_moves)
    opp = opponent_factory()

    try:
        w = d = l = 0
        t0 = time.time()
        for g in range(games):
            # alterna cores
            if g % 2 == 0:
                res = play_game(agent, opp)
                if res == "1-0":
                    w += 1
                elif res == "0-1":
                    l += 1
                else:
                    d += 1
            else:
                res = play_game(opp, agent)
                if res == "0-1":
                    w += 1
                elif res == "1-0":
                    l += 1
                else:
                    d += 1

            if (g + 1) % 50 == 0:
                rate = (g + 1) / max(1e-9, (time.time() - t0))
                print(
                    f"[{g+1}/{games}] {opponent_name} — W:{w} D:{d} L:{l} | {rate:.2f} g/s"
                )

        pts = w + 0.5 * d
        print(f"\n=== Resultado Final ({opponent_name}) ===")
        print(f"Vitórias: {w}  Empates: {d}  Derrotas: {l}  Pontos: {pts:.1f}/{games}")
    finally:
        # fechar engines
        if hasattr(agent, "close"):
            agent.close()
        if hasattr(opp, "close"):
            opp.close()


# ---------------- Main (F5) ---------------- #
if __name__ == "__main__":
    print(
        f"[INFO] Top-K={TOPK} | SF pick depth={SF_PICK_DEPTH} | Threads={SF_PICK_THREADS} | Hash={SF_PICK_HASH_MB}MB"
    )
    if RUN_SF_D4:
        run_matches_vs(
            "Stockfish d4", lambda: StockfishAgent(STOCKFISH_PATH, 6), GAMES_PER_OPP
        )
