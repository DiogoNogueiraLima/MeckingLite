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


# ───────────── UTIL: DEDUP E SEED ───────────── #
def dedup_merge(existing_list, new_list):
    """Mantém apenas 1 por FEN, preservando os já existentes."""
    seen = {d["fen"] for d in existing_list}
    added = 0
    for d in new_list:
        if d["fen"] not in seen:
            existing_list.append(d)
            seen.add(d["fen"])
            added += 1
    return existing_list, added


def set_seed(seed: int | None):
    if seed is not None:
        random.seed(seed)


# ───────────── SCORE E POSIÇÃO ───────────── #
def convert_mate_score(score):
    if score.is_mate():
        n = abs(score.mate())
        sign = 1 if score.mate() > 0 else -1
        return sign * (10000 - (n * 100))
    return score.score()


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
    return board, history[-history_size:]


# ───────────── NEW: HEURÍSTICAS EASY/MEDIUM/HARD ───────────── #
def material_score(board: chess.Board) -> int:
    """Soma de material simples para detectar finais."""
    vals = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    for p, v in vals.items():
        score += v * (
            len(board.pieces(p, chess.WHITE)) + len(board.pieces(p, chess.BLACK))
        )
    return score


def is_simple_endgame(board: chess.Board) -> bool:
    """Heurística barata: sem damas e pouco material total."""
    q = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if q > 0:
        return False
    return material_score(board) <= 14


def tag_position(board: chess.Board, top_moves, thresholds) -> str:
    """
    Classifica em easy/medium/hard com base em:
      - cp_eq_thr: |eval| <= thr → posição “equilibrada”
      - rich_min_legal: nº de lances legais mínimo para ser “rica”
      - tactic_gap_cp: gap entre 1º e 2º melhor lance para ser “tática” (hard)
    """
    cp_eq_thr = thresholds.get("cp_eq_thr", 80)
    rich_min_legal = thresholds.get("rich_min_legal", 25)
    tactic_gap_cp = thresholds.get("tactic_gap_cp", 180)

    # Avaliações (relative já vem do seu convert_mate_score(score.relative))
    evals = [m["score_cp"] for m in top_moves]
    evals_sorted = sorted(evals, reverse=True)
    cp_best = evals_sorted[0]
    cp_second = evals_sorted[1] if len(evals_sorted) > 1 else evals_sorted[0]
    gap = cp_best - cp_second

    legal_count = board.legal_moves.count()
    near_equal = abs(cp_best) <= cp_eq_thr
    big_gap = gap >= tactic_gap_cp

    if is_simple_endgame(board) and near_equal:
        return "easy"
    if legal_count >= rich_min_legal and near_equal:
        return "medium"
    if big_gap:
        return "hard"
    # Fallback razoável
    return "medium" if near_equal else "easy"


# ───────────── WORKER ───────────── #
def worker_task(args):
    # NEW: thresholds também chegam como argumento
    stockfish_path, depth, batch_size, thresholds = args
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
                        if len(uci) == 5:
                            if uci[-1].lower() != "q":
                                continue
                            move_cleaned = uci[:-1]
                        else:
                            move_cleaned = uci
                        top_moves.append(
                            {
                                "move": move_cleaned,
                                "score_cp": convert_mate_score(score.relative),
                            }
                        )
                if len(top_moves) >= 1:
                    # NEW: calcula a tag
                    tag = tag_position(board, top_moves, thresholds)
                    data.append(
                        {
                            "fen": board.fen(),
                            "history": history,
                            "top_moves": top_moves,
                            "tag": tag,  # NEW
                        }
                    )
            except Exception:
                continue
        engine.quit()
    except Exception as e:
        logging.error(f"Erro ao iniciar engine: {e}")
    return data


# ───────────── SAVE ───────────── #
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
    merged, added = dedup_merge(existing_data, data)
    with open(path, "wb") as f:
        pickle.dump(merged, f)
    return len(merged), added


# ───────────── GERAÇÃO POR FASE ───────────── #
def generate_dataset_for_phase(
    phase_name,
    depth,
    total_count,
    stockfish_path,
    output_dir,
    workers,
    batch_size,
    thresholds,
    seed=None,
):
    # NEW: vamos salvar por subpastas/tag
    base_dir = os.path.join(output_dir, phase_name)
    paths = {
        "easy": os.path.join(base_dir, "easy", "stockfish_data.pkl"),
        "medium": os.path.join(base_dir, "medium", "stockfish_data.pkl"),
        "hard": os.path.join(base_dir, "hard", "stockfish_data.pkl"),
    }

    logging.info(
        f"🚀 Fase {phase_name} — Depth: {depth} — Total desejado: {total_count}"
    )
    logging.info(f"📦 Salvando em subpastas: {paths}")
    set_seed(seed)

    total_saved = 0
    with mp.Pool(processes=workers) as pool:
        pbar = tqdm(total=total_count)

        while total_saved < total_count:
            tasks = [
                (stockfish_path, depth, batch_size, thresholds) for _ in range(workers)
            ]
            results = pool.map(worker_task, tasks)
            batch = [item for sublist in results for item in sublist]

            if not batch:
                logging.warning("⚠️ Nenhuma posição gerada neste ciclo. Continuando...")
                continue

            # NEW: agrupa por tag e salva em cada arquivo correspondente
            added_now = 0
            for tag in ("easy", "medium", "hard"):
                bucket = [d for d in batch if d.get("tag") == tag]
                if not bucket:
                    continue
                _, added = incremental_save(paths[tag], bucket)
                added_now += added

            if added_now == 0:
                # Pode ter sido tudo duplicado; ainda assim seguimos.
                pass

            total_saved += added_now
            # Atualiza a barra apenas com novas FENs
            pbar.update(added_now)

        pbar.close()

    logging.info(
        f"✅ Fase {phase_name} finalizada com ~{total_saved} novas posições (somando os 3 buckets)."
    )


def generate_two_phases(cfg, workers, batch, seed=None):
    stockfish_path = cfg["stockfish_path"]
    output_data_dir = cfg["output_data_dir"]
    thresholds = cfg["curriculum"]["thresholds"]  # NEW

    for phase_name in ["v1_depth6", "v2_depth12"]:
        phases = cfg["phases"][phase_name]
        depth = phases["depth"]
        count = phases["positions"]
        generate_dataset_for_phase(
            phase_name=phase_name,
            depth=depth,
            total_count=count,
            stockfish_path=stockfish_path,
            output_dir=output_data_dir,
            workers=workers,
            batch_size=batch,
            thresholds=thresholds,  # NEW
            seed=seed,
        )


# ───────────── MAIN ───────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3, help="Núcleos paralelos")
    parser.add_argument(
        "--batch", type=int, default=80, help="Tamanho do lote por núcleo"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="v1_depth6",
        choices=["v1_depth6", "v2_depth12", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    thresholds = cfg["curriculum"]["thresholds"]  # NEW

    if args.phase == "both":
        generate_two_phases(cfg, workers=args.workers, batch=args.batch, seed=args.seed)
    else:
        phases = cfg["phases"][args.phase]
        generate_dataset_for_phase(
            phase_name=args.phase,
            depth=phases["depth"],
            total_count=phases["positions"],
            stockfish_path=cfg["stockfish_path"],
            output_dir=cfg["output_data_dir"],  # NEW (veja YAML abaixo)
            workers=args.workers,
            batch_size=args.batch,
            thresholds=thresholds,  # NEW
            seed=args.seed,
        )
