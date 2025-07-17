import os
import pickle
import chess
from softmax import scores_to_policy_distribution

# ─────────────── CONFIG ─────────────── #
DATA_PATH = "data/v1_depth6/stockfish_data.pkl"
TEMPERATURE = 2.0
RESIDUAL = 0.01

# ─────────────── PROCESSAMENTO ─────────────── #


def process_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    updated_data = []

    for example in data:
        fen = example.get("fen")
        top_moves_raw = example.get("top_moves")

        if not fen or not top_moves_raw or len(top_moves_raw) < 3:
            updated_data.append(example)
            continue

        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        top_moves = []
        for move_info in top_moves_raw:
            move_uci = move_info.get("move")
            score_cp = move_info.get("score_cp")
            if move_uci is None or score_cp is None:
                continue
            top_moves.append((move_uci, score_cp))

        full_policy = scores_to_policy_distribution(
            top_moves, legal_moves, temperature=TEMPERATURE, residual=RESIDUAL
        )

        top_policy = {
            move: prob
            for move, prob in full_policy.items()
            if move in [m[0] for m in top_moves]
        }
        residual = sum(
            prob for move, prob in full_policy.items() if move not in top_policy
        )

        example["top_policy"] = top_policy
        example["residual"] = residual

        updated_data.append(example)

    with open(file_path, "wb") as f:
        pickle.dump(updated_data, f)

    print(f"✅ Dados atualizados com sucesso! Total de entradas: {len(updated_data)}")


if __name__ == "__main__":
    absolute_path = os.path.join(os.getcwd(), DATA_PATH)
    process_data(absolute_path)
