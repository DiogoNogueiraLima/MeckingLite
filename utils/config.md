1️⃣ cp_eq_thr: 80
Significado: “Centipawn Equal Threshold” → define o que é considerado posição equilibrada.

Como funciona: se a avaliação absoluta do Stockfish for <= 80 cp (0,80 peão), a posição é tratada como equilibrada.

Por que importa: posições equilibradas são mais “abertas” a múltiplos lances bons, ótimas para treinar discriminação e jogo posicional.

💡 Exemplo:

+0.45 (45 cp) → equilibrada.

+1.20 (120 cp) → já é vantagem clara, sai do bucket de “equilibrada”.

2️⃣ rich_min_legal: 25
Significado: “Richness = Mínimo de Lances Legais” → mede ramificação alta (quantos movimentos possíveis o lado a jogar tem).

Como funciona: se o número de lances legais for ≥ 25, a posição é considerada rica (muitos caminhos possíveis).

Por que importa:

Essas posições forçam o modelo a distinguir entre muitas opções quase equivalentes.

Melhora a capacidade da política de rankear corretamente entre vários lances bons.

💡 Exemplo:

Posição aberta, sem trocas → 32 lances possíveis → rica.

Posição travada, final de torres → 8 lances possíveis → não rica.

3️⃣ tactic_gap_cp: 180
Significado: “Gap Tático em Centipawns” → mede a diferença entre o melhor lance e o segundo melhor.

Como funciona: se o melhor lance for ≥ 180 cp melhor que o segundo melhor, consideramos posição tática/crítica (hard).

Por que importa:

Ensina a rede a evitar blunders e a identificar jogadas forçadas.

Muito útil para treinar precisão tática.

💡 Exemplo:

Melhor: +0.50, segundo: -1.50 → diferença = 200 cp → posição tática.

Melhor: +0.50, segundo: +0.35 → diferença = 15 cp → posição não tática.