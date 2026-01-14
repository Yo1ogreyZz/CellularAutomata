import os
from typing import Iterable, List
import torch
from torch_geometric.data import InMemoryDataset
from eca_lattice import cell_lattice_graph_data


class ECALatticeDataset(InMemoryDataset):
    """
    Each sample: rule-augmented cell-lattice graph.
    """
    def __init__(
        self,
        root: str,
        N: int = 16,
        r: int = 1,
        periodic: bool = True,
        # rule-aug params
        T: int = 8,
        num_seeds: int = 4,
        base_seed: int = 0,
        use_rule_bits: bool = True,
        rules: Iterable[int] = range(256),
        transform=None,
        pre_transform=None
    ):
        self.rules = list(rules)
        self.N = N
        self.r = r
        self.periodic = periodic
        self.T = T
        self.num_seeds = num_seeds
        self.base_seed = base_seed
        self.use_rule_bits = use_rule_bits

        super().__init__(root, transform, pre_transform)

        # PyTorch 2.6+ compatibility with PyG objects
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return ["placeholder.txt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        p = os.path.join(self.raw_dir, "placeholder.txt")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("placeholder")

    def process(self):
        data_list = [
            cell_lattice_graph_data(
                rn,
                N=self.N, r=self.r, periodic=self.periodic,
                T=self.T, num_seeds=self.num_seeds, base_seed=self.base_seed,
                use_rule_bits=self.use_rule_bits
            )
            for rn in self.rules
        ]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])