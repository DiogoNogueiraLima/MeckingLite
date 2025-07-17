import os
import pickle
import random
import matplotlib.pyplot as plt
import chess
import chess.svg
import chess.pgn
import io
from PIL import Image
import cairosvg

# caminho para o arquivo de dados gerado pelo Stockfish
PKL_PATH = r"C:\Users\diogo\Repositorios\MeckingLite\data\v1_depth6\stockfish_data.pkl"


# função auxiliar para converter pontuação tipo mate para centipawn
def score_to_cp(score):
    if isinstance(score, chess.engine.Mate):
        # mate positivo ou negativo, representado com grande valor
        return 3000 * (1 if score.mate() > 0 else -1)
    elif isinstance(score, chess.engine.Cp):
        return score.score()
    elif isinstance(score, dict) and "type" in score:
        # já convertido anteriormente
        if score["type"] == "mate":
            return 3000 * (1 if score["value"] > 0 else -1)
        else:
            return score["value"]
    else:
        return 0


# função para desenhar o tabuleiro com matplotlib
def display_board_with_info(fen, top_moves):
    board = chess.Board(fen)

    # define cores das 3 melhores jogadas
    colors = ["gold", "green", "blue"]
    arrows = []

    for i, move_info in enumerate(top_moves):
        move = chess.Move.from_uci(move_info["move"])
        arrows.append(
            chess.svg.Arrow(move.from_square, move.to_square, color=colors[i])
        )

    # gera o svg com as setas desenhadas
    svg_data = chess.svg.board(board=board, size=400, arrows=arrows)

    # converte para imagem para visualização no matplotlib
    png_data = cairosvg.svg2png(bytestring=svg_data)
    image = Image.open(io.BytesIO(png_data))

    # plota imagem com matplotlib e mostra jogadas ao lado
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # imagem do tabuleiro
    axs[0].imshow(image)
    axs[0].axis("off")
    axs[0].set_title("Tabuleiro")

    # texto com scores
    axs[1].axis("off")
    axs[1].set_title("Top 3 jogadas")
    for i, move_info in enumerate(top_moves):
        move_text = f"{i+1}. {move_info['move']} → {move_info['score_cp']} cp"
        axs[1].text(0.1, 1 - i * 0.2, move_text, fontsize=12, color=colors[i])

    plt.tight_layout()
    plt.show()


# carrega o arquivo pickle
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
    print("Quantidade de partidas no dataset: ", len(data))

# embaralha e seleciona N exemplos
N = 5
sampled = random.sample(data, N)

# visualiza cada um
for example in sampled:
    fen = example["fen"]
    top_moves = example[
        "top_moves"
    ]  # deve conter uma lista de dicts com "uci" e "score"
    print(f"\n📍 FEN: {fen}")
    display_board_with_info(fen, top_moves)
