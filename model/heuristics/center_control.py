import chess

MAIN_CENTRAL_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTRAL_SQUARES = [
    chess.C3,
    chess.C4,
    chess.C5,
    chess.C6,
    chess.F3,
    chess.F4,
    chess.F5,
    chess.F6,
]


def evaluate_center_control(board):
    white_main = white_extended = 0
    black_main = black_extended = 0

    for square in MAIN_CENTRAL_SQUARES:
        white_main += len(board.attackers(chess.WHITE, square))
        black_main += len(board.attackers(chess.BLACK, square))

    for square in EXTENDED_CENTRAL_SQUARES:
        white_extended += len(board.attackers(chess.WHITE, square))
        black_extended += len(board.attackers(chess.BLACK, square))

    main = white_main - black_main  # em [–8,+8]
    ext = white_extended - black_extended  # em [–16,+16]
    return main / 8.0, ext / 16.0
