"""FastAPI server to select chess moves using ChessPolicyNetwork.

Run with: ``uvicorn api.server:app``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.dataset import board_to_tensor, get_move_index_map
from model.heuristics.basic_heuristics import evaluate_all
from model.network import ChessPolicyNetwork


# ---------------------------------------------------------------------------
# Configuration and model loading
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "utils" / "config.yaml"

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f) or {}
else:
    CFG = {}

HISTORY_SIZE = int(CFG.get("history_size", 0))
USE_HEURISTICS = bool(CFG.get("use_heuristics", True))
CHECKPOINT_PATH = Path(
    CFG.get("checkpoint_path", ROOT / "checkpoints" / "supervised_epoch10.pt")
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

move_index_map, NUM_MOVES = get_move_index_map()
inv_move_index_map = {v: k for k, v in move_index_map.items()}

net = ChessPolicyNetwork(
    num_moves=NUM_MOVES,
    history_size=HISTORY_SIZE,
    heur_dim=8,
    use_heuristics=USE_HEURISTICS,
).to(DEVICE)
net.eval()

if CHECKPOINT_PATH.exists():
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        net.load_state_dict(state["model"], strict=False)
    else:
        net.load_state_dict(state, strict=False)
else:
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_inputs(board: chess.Board):
    board_np = board_to_tensor(board)
    if HISTORY_SIZE > 0:
        hist = np.zeros((12 * HISTORY_SIZE, 8, 8), dtype=np.float32)
        board_np = np.concatenate([hist, board_np], axis=0)
    board_tensor = torch.from_numpy(board_np).unsqueeze(0)

    heur = None
    if USE_HEURISTICS:
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


def legal_mask_for_board(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
    for mv in board.legal_moves:
        uci = mv.uci()
        if len(uci) == 5 and uci[-1].lower() == "q":
            uci = uci[:-1]
        idx = move_index_map.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def idx_to_uci4(idx: int) -> Optional[str]:
    return inv_move_index_map.get(idx)


def uci4_to_legal_move(board: chess.Board, uci4: str) -> Optional[chess.Move]:
    candidates = [mv for mv in board.legal_moves if mv.uci().startswith(uci4)]
    if not candidates:
        return None
    for mv in candidates:
        if mv.uci().endswith("q"):
            return mv
    return candidates[0]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


class MoveRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move: str


@app.post("/move", response_model=MoveResponse)
def select_move(req: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(req.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    legal_mask = legal_mask_for_board(board)
    bt, ht, hmt = build_inputs(board)
    bt, hmt = bt.to(DEVICE), hmt.to(DEVICE)
    if ht is not None:
        ht = ht.to(DEVICE)

    with torch.no_grad():
        logits = net(bt, ht, hmt)
        masked = logits.masked_fill(~legal_mask.to(DEVICE), float("-inf"))
        idx = int(torch.softmax(masked, dim=1).argmax(dim=1).item())

    uci4 = idx_to_uci4(idx)
    mv = uci4_to_legal_move(board, uci4) if uci4 else None
    if mv is None:
        raise HTTPException(status_code=400, detail="No legal move found")

    return MoveResponse(move=mv.uci())
