import torch
import chess
import chess.pgn
import numpy as np
from torch.utils.data import Dataset
from model.heuristics.basic_heuristics import evaluate_all


def board_to_tensor(board):
    """
    Converte um tabuleiro para um tensor 12x8x8.
    Cada camada representa uma peça específica para cada cor.
    """
    piece_map = board.piece_map()
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_to_index = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_to_index[piece.symbol()]
        tensor[idx][row][col] = 1.0

    return tensor


def get_move_index_map():
    move_index_map = {}
    files = "abcdefgh"
    ranks = "12345678"
    idx = 0
    for f1 in files:
        for r1 in ranks:
            for f2 in files:
                for r2 in ranks:
                    # pula movimentos sem alteração de casa
                    if f1 == f2 and r1 == r2:
                        continue
                    move = f1 + r1 + f2 + r2
                    move_index_map[move] = idx
                    idx += 1

    return move_index_map, len(move_index_map)


def board_and_history_to_tensor(history_fens, current_fen, history_size):
    """
    Recebe lista de FENs históricas e a FEN atual.
    Concatena até history_size históricos + posição atual,
    resultando em tensor de forma [12*(history_size+1), 8, 8].
    """
    # Últimos history_size históricos
    if history_size > 0:
        history = history_fens[-history_size:]
    else:
        history = []
    mats = []
    for fen in history:
        mats.append(board_to_tensor(chess.Board(fen)))
    # posição atual por último
    mats.append(board_to_tensor(chess.Board(current_fen)))
    if len(mats) > 1:
        return np.concatenate(mats, axis=0)
    else:
        return mats[0]


class ChessSupervisedDataset(Dataset):

    def __init__(self, data, use_heuristics=True, mode="top3", history_size=0):
        self.data = data
        self.use_heuristics = use_heuristics
        self.mode = mode
        self.history_size = history_size

        self.move_index_map, self.num_moves = get_move_index_map()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        history = sample.get("history", [])  # lista de FENs anteriores
        current_board = chess.Board(sample["fen"])

        # 0) Basic game info
        turn_flag = float(current_board.turn)
        half_moves = (current_board.fullmove_number - 1) * 2 + int(
            current_board.turn == chess.BLACK
        )
        half_mov_n_turn_tensor = torch.tensor(
            [half_moves / 40, turn_flag], dtype=torch.float32
        )

        # 1) Board Tensor
        board_hist_tensor = torch.tensor(
            board_and_history_to_tensor(history, sample["fen"], self.history_size),
            dtype=torch.float32,
        )

        # 2) Vetor de heurísticas
        if self.use_heuristics:
            mat, mob, ks, main_cc, ext_cc, white_bishops, black_bishops, total = (
                evaluate_all(current_board)
            )
            heuristic_tensor = torch.tensor(
                [mat, mob, ks, main_cc, ext_cc, white_bishops, black_bishops, total],
                dtype=torch.float32,
            )
        else:
            heuristic_tensor = torch.zeros(8, dtype=torch.float32)

        # 3) Política target
        policy = torch.zeros(self.num_moves, dtype=torch.float32)

        if self.mode == "top1":
            # One-hot na melhor jogada
            best_move = sample["top_moves"][0]["move"]
            idx_move = self.move_index_map.get(best_move)
            if idx_move is not None:
                policy[idx_move] = 1.0

        elif self.mode == "top3":
            # Reconstrói distribuição a partir de top_policy + residual_each
            top_policy = sample.get("top_policy", None)
            residual = float(sample.get("residual_each", 0.0))

            # Se não houver, faça fallback para top1
            if not top_policy:
                best_move = sample["top_moves"][0]["move"]
                idx_move = self.move_index_map.get(best_move)
                if idx_move is not None:
                    policy[idx_move] = 1.0
            else:
                # Preenche top-k
                for mv, prob in top_policy.items():
                    idx_move = self.move_index_map.get(mv)
                    if idx_move is not None:
                        policy[idx_move] = float(prob)

                # Residual para lances legais fora do top-k
                for move in current_board.legal_moves:
                    uci = move.uci()
                    if len(uci) == 5 and uci[-1].lower() == "q":
                        uci = uci[:-1]
                    if uci not in top_policy:
                        idx_move = self.move_index_map.get(uci)
                        if idx_move is not None:
                            policy[idx_move] = residual

                # Normaliza por segurança (deve somar ~1)
                s = policy.sum().item()
                if s > 0:
                    policy /= s

        # 4) Máscara de movimentos legais
        legal_mask = torch.zeros(self.num_moves, dtype=torch.bool)
        for move in current_board.legal_moves:
            uci = move.uci()
            # unificar promoções para dama
            if len(uci) == 5 and uci[-1].lower() == "q":
                uci = uci[:-1]
            idx_move = self.move_index_map.get(uci)
            if idx_move is not None:
                legal_mask[idx_move] = True

        return (
            half_mov_n_turn_tensor,
            board_hist_tensor,
            heuristic_tensor,
            policy,
            legal_mask,
        )


# ───────────── EXECUÇÃO PRINCIPAL ───────────── #

if __name__ == "__main__":
    import pickle
    from torch.utils.data import DataLoader

    # caminho de teste para o dataset gerado
    path = r"C:\Users\diogo\Repositorios\MeckingLite\data\v1_depth6\stockfish_data.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    # 2) Cria mapa de índices para todos os movimentos UCI de 4 caracteres
    dataset = ChessSupervisedDataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for half_mov_n_turn_tensor, board_hist_tensor, heur_tensor, policy in dataloader:
        print("📦 board_hist_tensor shape:", board_hist_tensor.shape)  # [B,12,8,8]
        print("📈 heuristic_tensor shape:", heur_tensor.shape)  # [B,6]
        print("🔢 policy shape:", policy.shape)  # [B, num_moves]
        print("Σ(policy[0]) =", policy[0].sum().item())  # deve ser ~1.0
        break
