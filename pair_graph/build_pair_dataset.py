from pair_dataset import ECAPairDataset

dataset = ECAPairDataset(root="./eca_pair_dataset", rules=range(256))
print("Saved:", len(dataset), "graphs")