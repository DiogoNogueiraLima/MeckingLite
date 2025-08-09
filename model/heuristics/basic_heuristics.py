from .material import evaluate_material
from .mobility import evaluate_mobility
from .king_safety import evaluate_king_safety
from .center_control import evaluate_center_control


def evaluate_all(board, weights=None):
    weights = weights or {
        "material": 1.0,
        "king_safety": 0.65,
        "mobility": 0.55,
        "main_center_control": 0.2,
        "extended_center_control": 0.1,
    }
    mat, white_bishops, black_bishops = evaluate_material(board)
    mob = evaluate_mobility(board)
    ks = evaluate_king_safety(board)
    main_cc, ext_cc = evaluate_center_control(board)

    total = 0

    total += weights["material"] * mat
    total += weights["mobility"] * mob
    total += weights["king_safety"] * ks
    total += weights["main_center_control"] * main_cc
    total += weights["extended_center_control"] * ext_cc

    soma_pesos = sum(weights.values())
    total_normalized = total / soma_pesos if soma_pesos != 0 else 0.0

    return mat, mob, ks, main_cc, ext_cc, white_bishops, black_bishops, total_normalized
