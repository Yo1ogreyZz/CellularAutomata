from stg_dataset import ECASTGDataset

dataset = ECASTGDataset(root="./eca_stg_N8_dataset", N=8, rules=range(256))
print("Saved:", len(dataset), "graphs")