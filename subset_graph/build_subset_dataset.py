from subset_dataset import ECASubsetDataset

dataset = ECASubsetDataset(root="./eca_subset_dataset", rules=range(256))
print("Saved:", len(dataset), "graphs")