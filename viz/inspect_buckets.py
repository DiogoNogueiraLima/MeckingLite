import pickle, os

for tag in ["easy", "medium", "hard"]:
    p = os.path.join("data", "v1_depth6", tag, "stockfish_data.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            d = pickle.load(f)
        print(tag, len(d))
        print(d[0])
    else:
        print(tag, 0)
