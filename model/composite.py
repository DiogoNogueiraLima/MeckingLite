import os
import pickle
import random
from torch.utils.data import Dataset
from model.dataset import ChessSupervisedDataset


def _load_bucket(phase_dir: str, tag: str):
    path = os.path.join(phase_dir, tag, "stockfish_data.pkl")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


class CompositeDataset(Dataset):
    """
    Mistura easy/medium/hard com pesos do currículo.
    Internamente mantém três ChessSupervisedDataset (um por bucket).
    Cada __getitem__ sorteia o bucket por probabilidade e devolve um item daquele bucket.
    """

    def __init__(
        self,
        phase_dir: str,
        *,
        history_size: int,
        use_heuristics: bool,
        mode: str,
        weights: dict,
    ):
        super().__init__()
        # carrega listas brutas
        self.raw = {
            "easy": _load_bucket(phase_dir, "easy"),
            "medium": _load_bucket(phase_dir, "medium"),
            "hard": _load_bucket(phase_dir, "hard"),
        }
        # cria os três datasets usando seu encoder atual
        self.ds = {
            k: ChessSupervisedDataset(
                v, use_heuristics=use_heuristics, mode=mode, history_size=history_size
            )
            for k, v in self.raw.items()
        }
        self.tags = ["easy", "medium", "hard"]
        self.set_weights(weights)

    def set_weights(self, weights: dict):
        """Troca pesos em tempo de execução (ex.: start → middle → end)."""
        w = {k: float(weights.get(k, 0.0)) for k in self.tags}
        s = sum(w.values())
        if s <= 0:
            w = {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}
            s = 1.0
        self.weights = {k: v / s for k, v in w.items()}
        # cumulativo para amostragem
        acc = 0.0
        self.cum = []
        for t in self.tags:
            acc += self.weights[t]
            self.cum.append(acc)

    def set_stage(self, schedule: dict, stage: str):
        if stage not in schedule:
            raise ValueError(f"Stage '{stage}' não encontrado no schedule.")
        self.set_weights(schedule[stage])

    def __len__(self):
        # tamanho "virtual": a soma dos buckets (apenas para satisfazer DataLoader)
        return sum(len(d) for d in self.ds.values())

    def _pick_tag(self) -> str:
        r = random.random()
        for t, c in zip(self.tags, self.cum):
            if r <= c:
                return t
        return self.tags[-1]

    def __getitem__(self, _):
        # tenta algumas vezes caso um bucket esteja vazio
        for _try in range(10):
            tag = self._pick_tag()
            d = self.ds[tag]
            if len(d) > 0:
                j = random.randrange(len(d))
                return d[j]  # retorna exatamente o 5-tuplo que seu treino espera
        raise RuntimeError("Todos os buckets estão vazios.")
