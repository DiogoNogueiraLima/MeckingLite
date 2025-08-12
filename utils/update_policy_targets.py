import os, pickle, chess, math, argparse


# ---------- helpers ----------
def _norm_uci4(u):
    return u[:4] if u and len(u) == 5 and u[-1].lower() in "qrbn" else u


def _softmax(scores, T):
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / T) for s in scores]
    Z = sum(exps)
    return [e / Z for e in exps] if Z > 0 else [1.0 / len(scores)] * len(scores)


def make_top_policy_and_residual(
    fen, top_moves, temperature_cp=300.0, topk=3, alpha=0.01
):
    """Retorna (top_policy: dict uci4->prob , residual_each: float)."""
    if not top_moves:
        return {}, 0.0
    # normaliza top-k
    tm = []
    for m in top_moves[:topk]:
        u, cp = m.get("move"), m.get("score_cp")
        if u is None or cp is None:
            continue
        tm.append((_norm_uci4(u), float(cp)))
    if not tm:
        return {}, 0.0

    # legais (normalizados)
    board = chess.Board(fen)
    legals = list({_norm_uci4(m.uci()): None for m in board.legal_moves}.keys())
    set_legals = set(legals)

    # softmax nos scores -> soma 1.0
    scores = [cp for (_, cp) in tm]
    probs = _softmax(scores, temperature_cp)

    # reescala para somar 0.99 (beta)
    beta = 1.0 - alpha
    probs = [p * beta for p in probs]  # somam ~0.99

    # filtra apenas lances legais nos top-k
    top_policy = {}
    for (u, _), p in zip(tm, probs):
        if u in set_legals:
            top_policy[u] = top_policy.get(u, 0.0) + p

    # residual distribuído igualmente entre legais fora do top-k
    others = [u for u in legals if u not in top_policy]
    residual_each = (alpha / len(others)) if others else 0.0
    return top_policy, float(residual_each)


# ---------- processamento ----------
def process_pkl(pkl_path, temperature_cp, topk, alpha, make_backup=True):
    if not os.path.exists(pkl_path):
        print(f"skip: {pkl_path} (não existe)")
        return
    print(f"Atualizando: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    updated, changed = [], 0
    for ex in data:
        fen = ex.get("fen")
        tms = ex.get("top_moves")
        if not fen or not tms:
            updated.append(ex)
            continue
        tp, residual_each = make_top_policy_and_residual(
            fen, tms, temperature_cp, topk, alpha
        )
        if tp:
            ex["top_policy"] = tp
            ex["residual_each"] = residual_each
            changed += 1
        updated.append(ex)

    if make_backup and not os.path.exists(pkl_path + ".bak"):
        with open(pkl_path + ".bak", "wb") as f:
            pickle.dump(data, f)
    with open(pkl_path, "wb") as f:
        pickle.dump(updated, f)
    print(f"  ✅ {changed}/{len(updated)} exemplos com top_policy + residual_each")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="raiz de dados (ex.: data)")
    ap.add_argument("--phases", nargs="+", default=["v1_depth6"])
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01)  # 1% para demais legais
    args = ap.parse_args()

    for ph in args.phases:
        for tag in ["easy", "medium", "hard"]:
            p = os.path.join(args.root, ph, tag, "stockfish_data.pkl")
            process_pkl(p, args.temperature, args.topk, args.alpha)


if __name__ == "__main__":
    main()
