#!/usr/bin/env python3
"""
Serviço de inferência que carrega o modelo a cada requisição.
Uso: python model/inference_service.py <fen>
Retorna: lance em formato UCI (ex: e2e4)
"""

import sys
import json
from pathlib import Path
import torch
import chess
import numpy as np
import yaml

# Imports do modelo
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import board_to_tensor, get_move_index_map
from model.heuristics.basic_heuristics import evaluate_all
from model.network import ChessPolicyNetwork


def load_config():
    """Carrega configuração do yaml"""
    root = Path(__file__).resolve().parents[1]
    config_path = root / "utils" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_inputs(board: chess.Board, history_size: int, use_heuristics: bool):
    """Constrói inputs para o modelo"""
    board_np = board_to_tensor(board)
    
    # Para compatibilidade com checkpoint: SEMPRE usar history_size = 4
    # Checkpoint espera 60 canais = 12 * (4 + 1)
    history_size = 4  # Forçar para compatibilidade
    hist = np.zeros((12 * history_size, 8, 8), dtype=np.float32)
    board_np = np.concatenate([hist, board_np], axis=0)
    
    board_tensor = torch.from_numpy(board_np).unsqueeze(0)

    heur = None
    if use_heuristics:
        mat, mob, ks, main_cc, ext_cc, wb, bb, total = evaluate_all(board)
        heur = torch.tensor(
            [[mat, mob, ks, main_cc, ext_cc, wb, bb, total]], dtype=torch.float32
        )

    turn_flag = float(board.turn)
    half_moves = (board.fullmove_number - 1) * 2 + int(board.turn == chess.BLACK)
    
    # Compatibilidade: usar apenas 2 dimensões como no checkpoint original
    half_moves_n_turn = torch.tensor(
        [[half_moves / 40.0, turn_flag]], dtype=torch.float32
    )
    
    return board_tensor, heur, half_moves_n_turn


def legal_mask_for_board(board: chess.Board, move_index_map, num_moves):
    """Cria máscara de lances legais"""
    mask = torch.zeros(num_moves, dtype=torch.bool)
    for mv in board.legal_moves:
        uci = mv.uci()
        if len(uci) == 5 and uci[-1].lower() == "q":
            uci = uci[:-1]
        idx = move_index_map.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask


def predict_move(fen: str):
    """
    Carrega o modelo, faz predição e retorna o lance.
    
    Args:
        fen: String FEN da posição
        
    Returns:
        lance UCI (str) ou None se erro
    """
    try:
        # 1. Carregar configuração
        cfg = load_config()
        history_size = int(cfg.get("history_size", 4))  # Forçar 4 para compatibilidade
        use_heuristics = bool(cfg.get("use_heuristics", True))
        
        root = Path(__file__).resolve().parents[1]
        checkpoint_path = Path(cfg.get("checkpoint_path", root / "checkpoints" / "supervised_epoch10.pt"))
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
        
        # 2. Criar board do chess.py
        board = chess.Board(fen)
        
        # Verificar se o jogo acabou
        if board.is_game_over():
            # Retornar informação sobre o final do jogo
            if board.is_checkmate():
                winner = "black" if board.turn == chess.WHITE else "white"
                return {"game_over": True, "result": "checkmate", "winner": winner}
            elif board.is_stalemate():
                return {"game_over": True, "result": "stalemate", "winner": "draw"}
            elif board.is_insufficient_material():
                return {"game_over": True, "result": "insufficient_material", "winner": "draw"}
            elif board.is_fivefold_repetition():
                return {"game_over": True, "result": "repetition", "winner": "draw"}
            elif board.is_seventyfive_moves():
                return {"game_over": True, "result": "75_moves", "winner": "draw"}
            else:
                return {"game_over": True, "result": "unknown", "winner": "draw"}
            
        # 3. Preparar dados do modelo
        move_index_map, num_moves = get_move_index_map()
        inv_move_index_map = {v: k for k, v in move_index_map.items()}
        
        # 4. Carregar modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        net = ChessPolicyNetwork(
            num_moves=num_moves,
            history_size=history_size,
            heur_dim=8,
            use_heuristics=use_heuristics,
            half_moves_n_turn_dim=2,  # Compatibilidade com checkpoint
            legacy_mode=True,  # ATIVA modo compatibilidade com checkpoint antigo
        ).to(device)
        
        # Carregar pesos
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            net.load_state_dict(state["model"], strict=False)
        else:
            net.load_state_dict(state, strict=False)
            
        net.eval()
        
        # 5. Preparar inputs
        board_tensor, heur_tensor, half_moves_tensor = build_inputs(board, history_size, use_heuristics)
        board_tensor = board_tensor.to(device)
        half_moves_tensor = half_moves_tensor.to(device)
        if heur_tensor is not None:
            heur_tensor = heur_tensor.to(device)
            
        # 6. Calcular máscara de lances legais
        legal_mask = legal_mask_for_board(board, move_index_map, num_moves).to(device)
        
        # 7. Fazer predição
        with torch.no_grad():
            logits = net(board_tensor, heur_tensor, half_moves_tensor)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            best_idx = int(torch.softmax(masked_logits, dim=1).argmax(dim=1).item())
            
        # 8. Converter índice para UCI
        uci4 = inv_move_index_map.get(best_idx)
        if uci4 is None:
            return None
            
        # 9. Encontrar lance legal correspondente
        candidates = [mv for mv in board.legal_moves if mv.uci().startswith(uci4)]
        if not candidates:
            return None
            
        # Preferir promoção para dama se houver
        for mv in candidates:
            if mv.uci().endswith("q"):
                return mv.uci()
                
        return candidates[0].uci()
        
    except Exception as e:
        print(f"Erro na predição: {e}", file=sys.stderr)
        return None


def main():
    """Função principal - recebe FEN via argv e retorna JSON"""
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python inference_service.py <fen>"}))
        sys.exit(1)
        
    fen = sys.argv[1]
    
    try:
        result = predict_move(fen)
        if result is None:
            print(json.dumps({"error": "No valid move found"}))
        elif isinstance(result, dict) and result.get("game_over"):
            # Final de jogo detectado
            print(json.dumps(result))
        elif isinstance(result, str):
            # Lance normal
            print(json.dumps({"move": result}))
        else:
            print(json.dumps({"error": "Invalid result from model"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
