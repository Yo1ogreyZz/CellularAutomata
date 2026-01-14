from dbg_dataset import ECADBGDataset

dataset = ECADBGDataset(root="./eca_dbg_dataset", rules=range(256))
print("Saved:", len(dataset), "graphs")