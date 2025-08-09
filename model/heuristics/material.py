import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.5,
    chess.BISHOP: 3.5,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def evaluate_material(board):
    bishops = board.piece_map().values()
    white_bishops = float(
        sum(
            1
            for p in bishops
            if p.piece_type == chess.BISHOP and p.color == chess.WHITE
        )
        == 2
    )
    black_bishops = float(
        sum(
            1
            for p in bishops
            if p.piece_type == chess.BISHOP and p.color == chess.BLACK
        )
        == 2
    )
    white_score = 0
    black_score = 0
    for piece_type in PIECE_VALUES:
        white_score += (
            len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        )
        black_score += (
            len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        )

    mat = white_score - black_score  # em [–39,+39]
    return mat / 39, white_bishops / 2, black_bishops / 2
