# utils/inspect_pkl_backup.py
import os
import pickle
import random
from collections import Counter

FILE_PATH = r"C:\Users\diogo\Repositorios\MeckingLite\data\v1_depth6\easy\stockfish_data.pkl.bak"


def short_dict(d, max_items=5):
    if not isinstance(d, dict):
        return d
    items = list(d.items())
    if len(items) <= max_items:
        return d
    head = items[:max_items]
    tail = len(items) - max_items
    return dict(head) | {"...": f"+{tail} more"}


def main():
    if not os.path.exists(FILE_PATH):
        print(f"Arquivo não encontrado:\n  {FILE_PATH}")
        return

    with open(FILE_PATH, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print("Erro ao ler pickle:", e)
            return

    n = len(data)
    print(f"\n✅ Carregado: {FILE_PATH}")
    print(f"Total de entradas: {n}")

    # estatísticas básicas
    has_top_policy = 0
    has_residual = 0
    len_top_moves = Counter()
    has_tag = 0

    for ex in data:
        tm = ex.get("top_moves", [])
        len_top_moves[len(tm)] += 1
        if "top_policy" in ex:
            has_top_policy += 1
        if "residual_each" in ex:
            has_residual += 1
        if "tag" in ex:
            has_tag += 1

    print("\nCampos presentes:")
    print(f"- Com 'top_policy':   {has_top_policy}  ({has_top_policy/n:.1%})")
    print(f"- Com 'residual_each':{has_residual}    ({has_residual/n:.1%})")
    print(f"- Com 'tag':          {has_tag}         ({has_tag/n:.1%})")

    # distribuição do tamanho de top_moves
    print("\nDistribuição do número de 'top_moves' por exemplo:")
    for k in sorted(len_top_moves):
        print(f"  {k}: {len_top_moves[k]}  ({len_top_moves[k]/n:.1%})")

    # amostras
    k_samples = min(3, n)
    print(f"\n=== {k_samples} exemplos aleatórios ===")
    for i, ex in enumerate(random.sample(data, k_samples), 1):
        fen = ex.get("fen", "<sem fen>")
        history = ex.get("history", [])
        tag = ex.get("tag", "<sem tag>")
        top_moves = ex.get("top_moves", [])
        tp = ex.get("top_policy", None)
        residual = ex.get("residual_each", None)

        print(f"\n[{i}] tag={tag}")
        print(f"FEN: {fen}")
        print(
            f"history (últimos {min(3, len(history))} de {len(history)}): {history[-3:] if history else []}"
        )

        # mostra até 5 top_moves
        show_tm = top_moves[:5]
        tm_fmt = ", ".join([f"{m.get('move')} ({m.get('score_cp')})" for m in show_tm])
        print(
            f"top_moves ({len(top_moves)}): {tm_fmt}{' ...' if len(top_moves)>5 else ''}"
        )

        if tp is not None:
            # imprime até 8 itens do top_policy
            print(f"top_policy (parcial): {short_dict(tp, max_items=8)}")
        else:
            print("top_policy: <não presente>")

        if residual is not None:
            print(f"residual_each: {residual:.6f}")
        else:
            print("residual_each: <não presente>")

    print(
        "\n📝 Dica: se quiser salvar um CSV/JSON de amostra, posso incluir no script."
    )


if __name__ == "__main__":
    main()
