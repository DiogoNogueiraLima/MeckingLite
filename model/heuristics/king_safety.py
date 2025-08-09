# king_safety.py
import chess
from model.heuristics.material import PIECE_VALUES

# Material máxima para normalização: queen(9)+2*rook(5)+2*bishop(3)+2*knight(3)+8*pawn(1)=39
MAX_MATERIAL_PER_SIDE = (
    PIECE_VALUES[chess.QUEEN]
    + 2 * PIECE_VALUES[chess.ROOK]
    + 2 * PIECE_VALUES[chess.BISHOP]
    + 2 * PIECE_VALUES[chess.KNIGHT]
    + 8 * PIECE_VALUES[chess.PAWN]
)


def evaluate_king_safety(board):
    """
    Avalia a segurança do rei, combinando:
      - Penalidade por estar no centro (d,e files)
      - Penalidade por não estar na fileira inicial (rank 0 para brancas, rank 7 para pretas)
      - Penalidade por ausência de peões de proteção diante do rei
    O impacto de todas as penalidades é ajustado de acordo com o material do adversário:
    quanto mais material o oponente tiver, maior a penalização (normalizado por MAX_MATERIAL_PER_SIDE).
    """
    total_score = 0.0

    # Função para calcular material do lado oposto
    def opponent_material(color):
        mat = 0
        for piece_type, value in PIECE_VALUES.items():
            count = len(board.pieces(piece_type, not color))
            mat += count * value
        return mat

    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue

        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)
        unscaled = 0.0

        # 1. Centro (files d=3, e=4)
        if file in (3, 4):
            unscaled += -2 if color == chess.WHITE else 2

        # 2. Fileira inicial (rank 0 para brancas, 7 para pretas)
        home_rank = 0 if color == chess.WHITE else 7
        if rank != home_rank:
            unscaled += -1.5 if color == chess.WHITE else 1.5

        # 3. Proteção de peões na frente do rei
        prot_rank = rank + 1 if color == chess.WHITE else rank - 1
        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f <= 7 and 0 <= prot_rank <= 7:
                sq = chess.square(f, prot_rank)
                if sq not in board.pieces(chess.PAWN, color):
                    unscaled += -0.5 if color == chess.WHITE else 0.5

        # Normaliza pela quantidade de material do oponente
        opp_mat = opponent_material(color)
        scale = opp_mat / MAX_MATERIAL_PER_SIDE if MAX_MATERIAL_PER_SIDE > 0 else 1.0
        total_score += unscaled * scale

    score = unscaled * (opp_mat / MAX_MATERIAL_PER_SIDE)

    return max(-1.0, min(1.0, score))
