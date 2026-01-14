from lattice_dataset import ECALatticeDataset

dataset = ECALatticeDataset(
    root="./eca_lattice_dataset_W16_T8_aug",
    N=16, r=1, periodic=True,
    T=8, num_seeds=4, base_seed=0,
    rules=range(256)
)

print("Saved:", len(dataset), "graphs")