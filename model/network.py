import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class ChessPolicyNetwork(nn.Module):
    def __init__(
        self,
        num_moves=4032,
        history_size=0,
        heur_dim=8,
        use_heuristics=True,
        half_moves_n_turn_dim=2,
        dropout: float = 0.0,  # NEW: opcional (padrão 0.0 = desativado)
    ):
        """
        num_moves: tamanho da política (nº de movimentos possíveis)
        history_size: quantas posições anteriores empilhar (12*(H+1) canais)
        heur_dim: dimensão do vetor de heurísticas
        use_heuristics: inclui ou não as heurísticas
        """
        super().__init__()
        self.half_moves_n_turn_dim = half_moves_n_turn_dim
        self.use_heuristics = use_heuristics
        in_channels = 12 * (history_size + 1)

        # Convs + BN + ReLU (iguais às suas, só com bias=False e ReLU inplace)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, padding=1, bias=False
        )  # NEW: bias=False
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1, bias=False
        )  # NEW: bias=False
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1, bias=False
        )  # NEW: bias=False
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(
            256, 256, kernel_size=3, padding=1, bias=False
        )  # NEW: bias=False
        self.bn4 = nn.BatchNorm2d(256)

        # Flatten + MLP
        self.flatten = nn.Flatten()
        self.fc_board = nn.Linear(256 * 8 * 8, 512)

        # Heurísticas (igual ao seu)
        if self.use_heuristics:
            self.fc_heur = nn.Linear(heur_dim, 32)
            combined_dim = 512 + 32
        else:
            self.fc_heur = None
            combined_dim = 512

        self.fc_combined = nn.Linear(combined_dim + self.half_moves_n_turn_dim, 256)
        self.dropout = nn.Dropout(dropout)  # NEW: simples e opcional (default 0.0)

        # Cabeça de política
        self.policy_head = nn.Linear(256, num_moves)

        # Inicialização de pesos (estável e rápida)
        self.apply(_init_weights)

    def forward(self, board_tensor, heur_tensor=None, half_moves_n_turn_tensor=None):
        # board_tensor: [B, in_channels, 8, 8]
        x = F.relu(self.bn1(self.conv1(board_tensor)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)

        x = self.flatten(x)  # [B, 256*8*8]
        x = F.relu(self.fc_board(x), inplace=True)  # [B, 512]

        if self.use_heuristics and heur_tensor is not None and self.fc_heur is not None:
            h = F.relu(self.fc_heur(heur_tensor), inplace=True)  # [B, 32]
            x = torch.cat([x, h], dim=1)  # [B, 544] se heurísticas

        if half_moves_n_turn_tensor is not None:
            # [B, 2] -> [halfmoves_norm, turn_flag]
            x = torch.cat([x, half_moves_n_turn_tensor], dim=1)

        x = F.relu(self.fc_combined(x), inplace=True)  # [B, 256]
        x = self.dropout(x)  # NEW: no-op se dropout=0.0
        policy_logits = self.policy_head(x)  # [B, num_moves]
        return policy_logits
