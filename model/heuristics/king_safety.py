# king_safety.py
import chess
from model.heuristics.material import PIECE_VALUES

MAX_MATERIAL_PER_SIDE = (
    PIECE_VALUES[chess.QUEEN]
    + 2 * PIECE_VALUES[chess.ROOK]
    + 2 * PIECE_VALUES[chess.BISHOP]
    + 2 * PIECE_VALUES[chess.KNIGHT]
    + 8 * PIECE_VALUES[chess.PAWN]
)


def _opp_material(board, color: chess.Color) -> float:
    mat = 0.0
    for pt, val in PIECE_VALUES.items():
        mat += len(board.pieces(pt, not color)) * val
    return mat


def evaluate_king_safety(board) -> float:
    """
    Score branco - preto, em ~[-1,1].
    Penaliza rei no centro, fora da rank inicial e sem escudo de peões,
    escalonado pelo material do oponente (quanto mais peças, maior o risco).
    """
    score_white = 0.0
    score_black = 0.0

    for color in (chess.WHITE, chess.BLACK):
        ksq = board.king(color)
        if ksq is None:
            continue

        rank = chess.square_rank(ksq)
        file = chess.square_file(ksq)
        unscaled = 0.0

        # 1) Rei no centro (arquivos d/e)
        if file in (3, 4):
            unscaled -= 2.0

        # 2) Fora da rank inicial
        home = 0 if color == chess.WHITE else 7
        if rank != home:
            unscaled -= 1.5

        # 3) Falta de peões no “escudo” (duas casas à frente do rei)
        prot_rank = rank + 1 if color == chess.WHITE else rank - 1
        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f <= 7 and 0 <= prot_rank <= 7:
                sq = chess.square(f, prot_rank)
                if sq not in board.pieces(chess.PAWN, color):
                    unscaled -= 0.5

        # Escalonar pelo material do oponente
        opp_mat = _opp_material(board, color)
        scale = opp_mat / MAX_MATERIAL_PER_SIDE if MAX_MATERIAL_PER_SIDE > 0 else 1.0
        s = unscaled * scale

        if color == chess.WHITE:
            score_white += s
        else:
            score_black += s

    # branco - preto e clamp ~[-1,1]
    raw = score_white - score_black
    return max(-1.0, min(1.0, raw))
