import os
import torch
from torch_geometric.data import InMemoryDataset
from eca_dependency import dependency_graph_data_with_rule


class DependencyRuleAugDataset(InMemoryDataset):
    def __init__(self, root: str,
                 W: int = 10, T: int = 8, r: int = 1, periodic: bool = True,
                 num_seeds: int = 4, base_seed: int = 0,
                 transform=None, pre_transform=None):
        self.W = W
        self.T = T
        self.r = r
        self.periodic = periodic
        self.num_seeds = num_seeds
        self.base_seed = base_seed
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []
        for rule in range(256):
            data = dependency_graph_data_with_rule(
                rule_number=rule,
                W=self.W, T=self.T, r=self.r, periodic=self.periodic,
                num_seeds=self.num_seeds, base_seed=self.base_seed
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])