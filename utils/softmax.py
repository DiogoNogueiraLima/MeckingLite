import numpy as np


def softmax(x, temperature=1.0):
    x = np.array(x, dtype=np.float32) / temperature
    e_x = np.exp(x - np.max(x))  # estabilidade numérica
    return e_x / e_x.sum()


def scores_to_policy_distribution(
    top_moves, legal_moves, temperature=2.0, residual=0.01
):
    """
    top_moves: lista de tuplas [(move_uci, score_cp)], onde score_cp é um int
    legal_moves: lista de strings UCI
    Retorna um dicionário {move_uci: probabilidade}
    """
    scores = np.array([cp / 100.0 for _, cp in top_moves])
    probs = softmax(scores, temperature)

    # monta dicionário inicial com as probabilidades dos top_moves
    policy = {move: prob for (move, _), prob in zip(top_moves, probs)}

    # distribui o residual entre os outros lances legais
    others = set(legal_moves) - set(policy)
    if others:
        r = residual / len(others)
        policy.update({move: r for move in others})

    # normaliza para garantir soma 1.0
    total = sum(policy.values())
    policy = {move: prob / total for move, prob in policy.items()}

    return policy


def example_usage():
    # Exemplo de uso com dummy
    legal = ["e2e4", "d2d4", "g1f3", "c2c4"]
    top3 = [
        ("e2e4", 543),
        ("d2d4", 438),
        ("g1f3", 399),
    ]

    policy = scores_to_policy_distribution(top3, legal, temperature=2.0)
    for move, prob in sorted(policy.items(), key=lambda x: -x[1]):
        print(f"{move}: {prob:.4f}")


if __name__ == "__main__":
    example_usage()
