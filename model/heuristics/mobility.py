import chess


def evaluate_mobility(board):
    """
    Calcula a diferença de mobilidade:
    número de jogadas legais para as brancas menos para as pretas.
    """
    # Mobilidade quando for a vez das brancas
    white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0

    # Simula o turno das pretas sem alterar de verdade o jogo
    board.push(chess.Move.null())
    black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()

    mob = white_mobility - black_mobility  # em algo até ±218
    return mob / 218.0
