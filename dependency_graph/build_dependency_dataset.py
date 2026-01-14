from dependency_dataset import DependencyRuleAugDataset

dataset = DependencyRuleAugDataset(
    root="./eca_dependency_dataset_W16_T8", W=16, T=8, r=1, periodic=True, num_seeds=4, base_seed=42
)

print("Saved:", len(dataset), "graphs")