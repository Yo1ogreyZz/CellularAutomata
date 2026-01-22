import torch

PT_FILE = "pyg_radius2_dependency_dedup.pt"

# node ids:
# 0..4 = input positions [-2,-1,0,+1,+2]
# 5    = output node
POS_NAMES = ["-2", "-1", "0", "+1", "+2", "out"]

def main():
    data_list = torch.load(PT_FILE, weights_only=False)

    print("Total graphs:", len(data_list))
    print("=" * 60)

    for i, data in enumerate(data_list[:5]):
        print(f"Graph #{i}")
        print(" rule_id:", data.rule_id)
        print(" y (label):", data.y.item())

        # shapes + dtypes
        print(" x shape:", tuple(data.x.shape), "dtype:", data.x.dtype)
        print(" edge_index shape:", tuple(data.edge_index.shape), "dtype:", data.edge_index.dtype)
        print(" edge_attr shape:", tuple(data.edge_attr.shape), "dtype:", data.edge_attr.dtype)

        # strict sanity checks for dependency representation
        assert tuple(data.x.shape) == (6, 2), "Expected x shape (6,2) for dependency graph"
        assert tuple(data.edge_index.shape)[0] == 2, "edge_index should have shape (2, E)"
        assert data.edge_index.dtype == torch.long, "edge_index must be torch.long"
        assert data.y.dtype == torch.long, "y must be torch.long"

        # edge_attr can be empty if E=0
        E = data.edge_index.shape[1]
        assert tuple(data.edge_attr.shape) == (E, 1), "edge_attr should have shape (E,1)"
        assert data.edge_attr.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), "edge_attr should be integer"

        # print nodes (all 6, small)
        print(" x (all nodes):")
        for nid in range(6):
            # x is one-hot type: [1,0]=input, [0,1]=output
            print(f"  node {nid} ({POS_NAMES[nid]}): x={data.x[nid].tolist()}")

        # print edges (dependency edges)
        print(" edges (dependency):")
        if E == 0:
            print("  (no dependency edges)")
        else:
            for e in range(min(5, E)):
                s = data.edge_index[0, e].item()
                t = data.edge_index[1, e].item()
                val = int(data.edge_attr[e, 0].item())  # should be 1
                print(f"  {s}({POS_NAMES[s]}) -> {t}({POS_NAMES[t]}), edge_attr={val}")

            if E > 5:
                print(f"  ... ({E-5} more edges)")

        # extra readable summary: which positions are dependent
        deps = []
        for e in range(E):
            s = data.edge_index[0, e].item()
            t = data.edge_index[1, e].item()
            if t == 5:
                deps.append(POS_NAMES[s])
        deps = sorted(deps, key=lambda x: ["-2","-1","0","+1","+2"].index(x) if x in ["-2","-1","0","+1","+2"] else 99)

        print(" dependent positions:", deps if deps else "[]")
        print("-" * 60)

if __name__ == "__main__":
    main()