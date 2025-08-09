import os
import yaml
import pickle
import random
import logging
import argparse
import multiprocessing as mp

import chess
import chess.engine
from tqdm import tqdm

# ───────────────────────────── CONFIG ──────────────────────────── ─#


def load_config(path="utils/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ───────────────────────────── LOGGING ──────────────────────────── ─#

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)

# ───────────── GERAÇÃO E AVALIAÇÃO ───────────── #


def convert_mate_score(score):
    if score.is_mate():
        # mate in N: quanto mais próximo do mate, mais extremo o valor
        n = abs(score.mate())
        sign = 1 if score.mate() > 0 else -1
        return sign * (10000 - (n * 100))  # exemplo: mate em 1 = 9900 cp
    return score.score()  # caso centipawn normal


# GERA POSIÇÃO E HISTÓRICO
def generate_random_position_with_history(min_plies=6, max_plies=40, history_size=4):
    board = chess.Board()
    history = []
    plies = random.randint(min_plies, max_plies)
    for _ in range(plies):
        if board.is_game_over():
            break
        history.append(board.fen())
        move = random.choice(list(board.legal_moves))
        board.push(move)
    # manter apenas os últimos N históricos
    return board, history[-history_size:]


# WORKER TASK
def worker_task(args):
    stockfish_path, depth, batch_size = args
    data = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        for _ in range(batch_size):
            board, history = generate_random_position_with_history()
            try:
                result = engine.analyse(
                    board, chess.engine.Limit(depth=depth), multipv=3
                )
                top_moves = []
                for r in result:
                    move = r.get("pv", [None])[0]
                    score = r.get("score")
                    if move and score:
                        uci = move.uci()
                        # Se for promoção (len 5), só aceite 'q' e remova o último char
                        if len(uci) == 5:
                            if uci[-1].lower() != "q":
                                continue
                            move_cleaned = uci[:-1]  # CORREÇÃO AQUI
                        else:
                            move_cleaned = uci
                        top_moves.append(
                            {
                                "move": move_cleaned,
                                "score_cp": convert_mate_score(score.relative),
                            }
                        )
                if len(top_moves) >= 1:
                    data.append(
                        {"fen": board.fen(), "history": history, "top_moves": top_moves}
                    )
            except Exception:
                continue
        engine.quit()
    except Exception as e:
        logging.error(f"Erro ao iniciar engine: {e}")
    return data


# ───────────── SALVAMENTO INCREMENTAL ───────────── #


def incremental_save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                existing_data = pickle.load(f)
        except Exception:
            existing_data = []
    else:
        existing_data = []
    all_data = existing_data + data
    with open(path, "wb") as f:
        pickle.dump(all_data, f)
    return len(all_data)


# ───────────── EXECUÇÃO POR FASE ───────────── #


def generate_dataset_for_phase(
    phase_name, depth, total_count, stockfish_path, output_dir, workers, batch_size
):
    output_path = os.path.join(output_dir, phase_name, "stockfish_data.pkl")
    logging.info(f"🚀 Fase {phase_name} — Depth: {depth} — Total: {total_count}")
    logging.info(f"📦 Salvando em: {output_path}")

    with mp.Pool(processes=workers) as pool:
        pbar = tqdm(total=total_count)
        total_generated = 0

        while total_generated < total_count:
            tasks = [(stockfish_path, depth, batch_size) for _ in range(workers)]
            results = pool.map(worker_task, tasks)
            batch_data = [item for sublist in results for item in sublist]

            if not batch_data:
                logging.warning("⚠️ Nenhuma posição gerada neste ciclo. Continuando...")
                continue

            total_generated = incremental_save(output_path, batch_data)
            pbar.update(len(batch_data))
        pbar.close()

    logging.info(f"✅ Fase {phase_name} finalizada com {total_generated} posições.")


# ───────────── EXECUÇÃO PRINCIPAL ───────────── #
PHASE_CHOOSEN = "v1_depth6"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3, help="Núcleos paralelos")
    parser.add_argument(
        "--batch", type=int, default=80, help="Tamanho do lote por núcleo"
    )
    args = parser.parse_args()

    cfg = load_config()
    stockfish_path = cfg["stockfish_path"]
    output_data_dir = cfg["output_data_dir"]
    phases = cfg["phases"][PHASE_CHOOSEN]

    depth = phases["depth"]
    count = phases["positions"]
    generate_dataset_for_phase(
        phase_name=PHASE_CHOOSEN,
        depth=depth,
        total_count=count,
        stockfish_path=stockfish_path,
        output_dir=output_data_dir,
        workers=args.workers,
        batch_size=args.batch,
    )
