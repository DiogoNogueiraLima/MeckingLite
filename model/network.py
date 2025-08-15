import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        # compatível caso existam BN antigas em checkpoints
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
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
        half_moves_n_turn_dim=4,  # [halfmoves_norm, turn, canK, canQ]
        dropout: float = 0.1,
        use_meta_planes: bool = True,  # NEW: injeta turn/castling como planos 8x8
        legacy_mode: bool = False,  # NEW: compatibilidade com checkpoint antigo
    ):
        super().__init__()
        self.use_heuristics = use_heuristics
        self.half_moves_n_turn_dim = half_moves_n_turn_dim
        self.use_meta_planes = use_meta_planes
        self.legacy_mode = legacy_mode

        if legacy_mode:
            # MODO LEGADO: compatível com checkpoint antigo
            # Força configurações específicas para checkpoint
            self.history_size = 4  # Fixo para checkpoint
            self.half_moves_n_turn_dim = 2  # Apenas halfmoves + turn
            self.use_meta_planes = False  # Sem meta planes no modelo antigo
            
            in_channels = 60  # 12 * (4 + 1) = 60 canais fixos
            
            # Arquitetura do checkpoint antigo
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(256)
            
            # FC para board features
            self.fc_board = nn.Linear(256 * 8 * 8, 512)
            self.gap = None  # Não usa GAP no modo legado
            
            # FC para heurísticas
            if self.use_heuristics:
                self.fc_heur = nn.Linear(heur_dim, 32)
            else:
                self.fc_heur = None
                
            # FC combinada: 512 + 32 + 2 = 546 -> 256
            combined_dim = 512 + (32 if self.use_heuristics else 0) + 2
            self.fc_combined = nn.Linear(combined_dim, 256)
            
        else:
            # MODO NOVO: arquitetura moderna
            base_in = 12 * (history_size + 1)
            # se usar planos, adicionamos 3 canais (turn, canK, canQ) antes das convs
            in_channels = base_in + (3 if self.use_meta_planes else 0)

            def GN(c):
                g = 8 if c >= 8 else 1
                return nn.GroupNorm(g, c)

            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
            self.gn1 = GN(64)
            self.conv2 = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
            self.gn2 = GN(96)
            self.conv3 = nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False)
            self.gn3 = GN(128)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
            self.gn4 = GN(128)

            self.gap = nn.AdaptiveAvgPool2d(1)
            feat_dim = 128

            if self.use_heuristics:
                self.fc_heur = nn.Linear(heur_dim, 16)
                combined_dim = feat_dim + 16
            else:
                self.fc_heur = None
                combined_dim = feat_dim

            # Mesmo que metade entre como planos, mantemos o vetor de 4 dims;
            # no forward vamos DESCARTAR turn/castling dessa parte para evitar duplicação.
            self.fc_combined = nn.Linear(
                combined_dim
                + self.half_moves_n_turn_dim
                - (3 if self.use_meta_planes else 0),
                128,
            )

        self.dropout = nn.Dropout(dropout)
        self.policy_head = nn.Linear(256 if legacy_mode else 128, num_moves)

        self.apply(_init_weights)

    def forward(self, board_tensor, heur_tensor=None, half_moves_n_turn_tensor=None):
        if self.legacy_mode:
            # FORWARD MODO LEGADO (compatível com checkpoint)
            x = F.relu(self.bn1(self.conv1(board_tensor)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            
            # Flatten para FC
            x = x.view(x.size(0), -1)  # [B, 256*8*8]
            x = F.relu(self.fc_board(x))  # [B, 512]
            
            # Heurísticas
            if self.use_heuristics and heur_tensor is not None and self.fc_heur is not None:
                h = F.relu(self.fc_heur(heur_tensor))  # [B, 32]
                x = torch.cat([x, h], dim=1)
                
            # Metadados (apenas halfmoves + turn no modo legado)
            if half_moves_n_turn_tensor is not None:
                meta = half_moves_n_turn_tensor[:, :2]  # [B, 2]
                x = torch.cat([x, meta], dim=1)
                
        else:
            # FORWARD MODO NOVO (arquitetura moderna)
            # half_moves_n_turn_tensor: [B, 4] -> [halfmoves_norm, turn, canK, canQ]
            if self.use_meta_planes and half_moves_n_turn_tensor is not None:
                B, _, H, W = board_tensor.shape
                # extrai turn/canK/canQ e expande para 8x8
                turn = half_moves_n_turn_tensor[:, 1].view(B, 1, 1, 1).expand(B, 1, H, W)
                canK = half_moves_n_turn_tensor[:, 2].view(B, 1, 1, 1).expand(B, 1, H, W)
                canQ = half_moves_n_turn_tensor[:, 3].view(B, 1, 1, 1).expand(B, 1, H, W)
                board_tensor = torch.cat([board_tensor, turn, canK, canQ], dim=1)

            x = F.relu(self.gn1(self.conv1(board_tensor)))
            x = F.relu(self.gn2(self.conv2(x)))
            x = F.relu(self.gn3(self.conv3(x)))
            x = F.relu(self.gn4(self.conv4(x)))

            x = self.gap(x).squeeze(-1).squeeze(-1)  # [B, 128]

            if self.use_heuristics and heur_tensor is not None and self.fc_heur is not None:
                h = F.relu(self.fc_heur(heur_tensor))  # [B, 16]
                x = torch.cat([x, h], dim=1)

            if half_moves_n_turn_tensor is not None:
                if self.use_meta_planes:
                    # só concatenamos o halfmoves_norm (índice 0) — 1 dimensão
                    x = torch.cat([x, half_moves_n_turn_tensor[:, 0:1]], dim=1)
                else:
                    # concatena todas as 4 dims no caso sem planos
                    x = torch.cat([x, half_moves_n_turn_tensor], dim=1)

        x = F.relu(self.fc_combined(x))
        x = self.dropout(x)
        return self.policy_head(x)
