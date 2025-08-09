import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessPolicyNetwork(nn.Module):
    def __init__(
        self,
        num_moves=4032,
        history_size=0,
        heur_dim=8,
        use_heuristics=True,
        half_moves_n_turn_dim=2,
    ):
        """
        num_moves: número de movimentos possíveis (tamanho da política)
        history_size: quantas posições anteriores foram empilhadas
        heur_dim: dimensão do vetor de heurísticas
        use_heuristics: se inclui ou não input heurístico
        """
        super().__init__()
        self.half_moves_n_turn_dim = half_moves_n_turn_dim
        self.use_heuristics = use_heuristics
        in_channels = 12 * (history_size + 1)

        # Camadas convolucionais com BatchNorm e ReLU
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Flatten para fully connected
        self.flatten = nn.Flatten()
        self.fc_board = nn.Linear(256 * 8 * 8, 512)

        # Ramificação para heurísticas
        if self.use_heuristics:
            self.fc_heur = nn.Linear(heur_dim, 32)
            combined_dim = 512 + 32
        else:
            combined_dim = 512
        self.fc_combined = nn.Linear(combined_dim + self.half_moves_n_turn_dim, 256)

        # Policy head
        self.policy_head = nn.Linear(256, num_moves)

    def forward(self, board_tensor, heur_tensor=None, half_moves_n_turn_tensor=None):
        # board_tensor: [B, in_channels, 8,8]
        x = F.relu(self.bn1(self.conv1(board_tensor)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)  # [B, 256*8*8]
        x = F.relu(self.fc_board(x))  # [B, 512]

        # Integra heurísticas se houver
        if self.use_heuristics and heur_tensor is not None:
            h = F.relu(self.fc_heur(heur_tensor))  # [B, 32]
            x = torch.cat([x, h], dim=1)  # [B, 544]

        if half_moves_n_turn_tensor is not None:
            # half_moves_n_turn_tensor: [B, 2] -> [halfmoves, turn]
            x = torch.cat([x, half_moves_n_turn_tensor], dim=1)

        x = F.relu(self.fc_combined(x))  # [B, 256]
        policy_logits = self.policy_head(x)  # [B, num_moves]
        return policy_logits
