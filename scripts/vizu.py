import pickle

# 1) Abra em modo binário de leitura
with open(
    r"C:\Users\diogo\Repositorios\MeckingLite\data\v1_depth6\stockfish_data.pkl", "rb"
) as f:
    obj = pickle.load(f)

# 2) Veja o tipo do objeto
print(type(obj))

# 3) Se for um dicionário, por exemplo:
if isinstance(obj, dict):
    print("Chaves:", obj.keys())
    # mostrar os primeiros 3 itens
    for k in list(obj.keys())[:3]:
        print(k, "→", obj[k])

# 4) Se for uma lista:
if isinstance(obj, list):
    print("Comprimento:", len(obj))
    print("Primeiros elementos:", obj[:3])
