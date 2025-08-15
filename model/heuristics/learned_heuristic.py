# model/heuristics/learned_heuristic.py
from __future__ import annotations
from pathlib import Path
import yaml
import chess

# Reaproveita suas features antigas (normalizadas)
from model.heuristics.basic_heuristics import evaluate_all as _eval_basic

_DEFAULT = {
    # pesos das features já existentes (todas são "branco - preto" normalizadas)
    "material": 1.00,
    "mobility": 0.20,
    "king_safety": 0.25,
    "main_center_control": 0.15,
    "extended_center_control": 0.05,
    "white_bishops": 0.02,
    "black_bishops": 0.02,  # manter explícito p/ tuning (pode até ser negativo)
    "bias": 0.0,
    "scale_cp": 100.0,  # multiplicador para aproximar centipawns
    # novas contribuições
    "center_phase_boost": 0.40,  # centro vale mais no início (modulado por fase)
    "center_piece_boost": 0.30,  # centro ponderado por nº de peças em jogo
    "rook_open_file": 0.15,
    "rook_semi_open_file": 0.08,
    "rook_on_7th": 0.20,
    "passed_pawn": 0.12,  # por “nível” de avanço (0..6) diferencial (W-B)
    "doubled_pawn_penalty": -0.10,
    "isolated_pawn_penalty": -0.12,
    "king_shield": 0.08,
}

_CACHED = dict(_DEFAULT)


def load_weights(yaml_path: Path) -> None:
    global _CACHED
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _CACHED = {**_DEFAULT, **data}
    else:
        _CACHED = dict(_DEFAULT)


def get_weights() -> dict:
    return dict(_CACHED)


# ---------------- features extras ----------------
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
EXT_CENTER = [
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.D4,
    chess.E4,
    chess.F4,
    chess.C5,
    chess.D5,
    chess.E5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
]


def _att_ctrl(board: chess.Board, squares, color: chess.Color) -> int:
    # board.attackers -> SquareSet
    return sum(len(board.attackers(color, sq)) for sq in squares)


def _center_control_raw(board: chess.Board):
    w = _att_ctrl(board, CENTER, chess.WHITE)
    b = _att_ctrl(board, CENTER, chess.BLACK)
    return float(w - b)


def _ext_center_control_raw(board: chess.Board):
    w = _att_ctrl(board, EXT_CENTER, chess.WHITE)
    b = _att_ctrl(board, EXT_CENTER, chess.BLACK)
    return float(w - b)


def _non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    # board.pieces -> SquareSet; use len(...)
    vals = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3}
    return sum(len(board.pieces(pt, color)) * v for pt, v in vals.items())


def _phase(board: chess.Board) -> float:
    # 62 aprox: 2Q(18)+4R(20)+4B(12)+4N(12)
    npm = _non_pawn_material(board, chess.WHITE) + _non_pawn_material(
        board, chess.BLACK
    )
    return 1.0 - min(1.0, npm / 62.0)  # 0 = abertura, 1 = final


def _file_is_open(board: chess.Board, f: int) -> bool:
    mask = chess.BB_FILES[f]  # int (bitboard)
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(
        chess.PAWN, chess.BLACK
    )  # SquareSet
    return len(pawns & mask) == 0


def _file_is_semi_open(board: chess.Board, f: int, color: chess.Color) -> bool:
    mask = chess.BB_FILES[f]
    my = board.pieces(chess.PAWN, color)
    op = board.pieces(chess.PAWN, not color)
    return len(my & mask) == 0 and len(op & mask) != 0


def _rook_on_7th(board: chess.Board, color: chess.Color) -> int:
    rank7 = chess.BB_RANK_7 if color == chess.WHITE else chess.BB_RANK_2
    rooks = board.pieces(chess.ROOK, color)  # SquareSet
    return len(rooks & rank7)


def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    cnt = 0
    p = board.pieces(chess.PAWN, color)  # SquareSet
    for f in range(8):
        k = len(p & chess.BB_FILES[f])
        if k > 1:
            cnt += k - 1
    return cnt


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    cnt = 0
    p = board.pieces(chess.PAWN, color)  # SquareSet
    for f in range(8):
        mask = chess.BB_FILES[f]
        if len(p & mask) > 0:
            left = chess.BB_FILES[f - 1] if f - 1 >= 0 else 0
            right = chess.BB_FILES[f + 1] if f + 1 < 8 else 0
            if len(p & left) == 0 and len(p & right) == 0:
                cnt += len(p & mask)
    return cnt


def _passed_pawn_adv(board: chess.Board, color: chess.Color) -> int:
    score = 0
    my = board.pieces(chess.PAWN, color)  # SquareSet (iterável de squares)
    opp = board.pieces(chess.PAWN, not color)
    for sq in my:
        f = chess.square_file(sq)
        files = {f}
        if f - 1 >= 0:
            files.add(f - 1)
        if f + 1 < 8:
            files.add(f + 1)
        mask = 0
        for ff in files:
            mask |= chess.BB_FILES[ff]
        if color == chess.WHITE:
            ahead = 0
            for r in range(chess.square_rank(sq) + 1, 8):
                ahead |= mask & chess.BB_RANKS[r]
        else:
            ahead = 0
            for r in range(0, chess.square_rank(sq)):
                ahead |= mask & chess.BB_RANKS[r]
        # opp & ahead -> SquareSet; vazio se len(...)==0
        if len(opp & ahead) == 0:
            rank = (
                chess.square_rank(sq)
                if color == chess.WHITE
                else (7 - chess.square_rank(sq))
            )
            score += rank  # 0..6
    return score


# ---------------- API compatível + total reponderado ----------------
def evaluate_all(board, weights=None):
    # antigas (normalizadas): mat, mob, ks, main, ext, wb, bb, total_norm_antigo
    mat, mob, ks, main_cc, ext_cc, wb, bb, _ = _eval_basic(board)
    w = {**_CACHED, **(weights or {})}

    # novas (diferenciais branco - preto)
    try:
        # Caso a sua versão retorne SquareSet
        pc = len(board.occupied)
    except TypeError:
        # Caso retorne int (bitboard); use popcount/bit_count
        try:
            import chess

            pc = chess.popcount(board.occupied)
        except Exception:
            pc = int(board.occupied).bit_count()
    # SquareSet -> use len(...)
    center_raw = _center_control_raw(board)
    center_phase = center_raw * (1.0 - _phase(board))
    center_piece = center_raw * (pc / 32.0)

    open_w = sum(
        _file_is_open(board, f)
        for f in range(8)
        if len(board.pieces(chess.ROOK, chess.WHITE) & chess.BB_FILES[f]) > 0
    )
    open_b = sum(
        _file_is_open(board, f)
        for f in range(8)
        if len(board.pieces(chess.ROOK, chess.BLACK) & chess.BB_FILES[f]) > 0
    )
    rook_open = float(open_w - open_b)

    semi_w = sum(
        _file_is_semi_open(board, f, chess.WHITE)
        for f in range(8)
        if len(board.pieces(chess.ROOK, chess.WHITE) & chess.BB_FILES[f]) > 0
    )
    semi_b = sum(
        _file_is_semi_open(board, f, chess.BLACK)
        for f in range(8)
        if len(board.pieces(chess.ROOK, chess.BLACK) & chess.BB_FILES[f]) > 0
    )
    rook_semi = float(semi_w - semi_b)

    rook7 = float(_rook_on_7th(board, chess.WHITE) - _rook_on_7th(board, chess.BLACK))
    passed = float(
        _passed_pawn_adv(board, chess.WHITE) - _passed_pawn_adv(board, chess.BLACK)
    )
    doubled = float(
        _doubled_pawns(board, chess.WHITE) - _doubled_pawns(board, chess.BLACK)
    )
    isolated = float(
        _isolated_pawns(board, chess.WHITE) - _isolated_pawns(board, chess.BLACK)
    )

    # “escudo” simples (peões à frente do rei, 2 ranks)
    def _king_shield(board, color):
        k = board.king(color)
        if k is None:
            return 0
        r = chess.square_rank(k)
        f = chess.square_file(k)
        files = {f}
        if f - 1 >= 0:
            files.add(f - 1)
        if f + 1 < 8:
            files.add(f + 1)
        ranks = (
            {min(r + 1, 7), min(r + 2, 7)}
            if color == chess.WHITE
            else {max(r - 1, 0), max(r - 2, 0)}
        )
        cnt = 0
        for rr in ranks:
            for ff in files:
                sq = chess.square(ff, rr)
                p = board.piece_at(sq)
                if p and p.color == color and p.piece_type == chess.PAWN:
                    cnt += 1
        return cnt

    kshield = float(_king_shield(board, chess.WHITE) - _king_shield(board, chess.BLACK))

    # repondera o total (sem normalizar no final — quem calibra é o tuner + scale_cp)
    total = (
        w["material"] * mat
        + w["mobility"] * mob
        + w["king_safety"] * ks
        + w["main_center_control"] * main_cc
        + w["extended_center_control"] * ext_cc
        + w["white_bishops"] * wb
        + w["black_bishops"] * bb
        + w["center_phase_boost"] * center_phase
        + w["center_piece_boost"] * center_piece
        + w["rook_open_file"] * rook_open
        + w["rook_semi_open_file"] * rook_semi
        + w["rook_on_7th"] * rook7
        + w["passed_pawn"] * passed
        + w["doubled_pawn_penalty"] * doubled
        + w["isolated_pawn_penalty"] * isolated
        + w["king_shield"] * kshield
        + w["bias"]
    )

    return mat, mob, ks, main_cc, ext_cc, wb, bb, float(total)


def score_cp_side_to_move(board) -> float:
    *_, total = evaluate_all(board)
    scale = float(_CACHED.get("scale_cp", 100.0))
    cp_white = float(total) * scale
    return cp_white if board.turn else -cp_white
