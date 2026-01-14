import os
from typing import Iterable, List
from torch_geometric.data import InMemoryDataset
from eca_stg import eca_stg_data

class ECASTGDataset(InMemoryDataset):
    """
    Each sample: STG of one ECA rule for a fixed N.
    Saved as root/processed/data.pt
    """
    def __init__(self, root: str, N: int = 8, rules: Iterable[int] = range(256),
                 transform=None, pre_transform=None):
        self.rules = list(rules)
        self.N = N
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

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
        data_list = [eca_stg_data(r, N=self.N) for r in self.rules]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        self.save(data_list, self.processed_paths[0])