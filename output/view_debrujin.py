
import torch

PT_FILE = "pyg_radius2_debruijn_dedup.pt"

def node_id_to_bits4(n: int):
    """0..15 -> [b0,b1,b2,b3]"""
    return [(n >> k) & 1 for k in reversed(range(4))]

def main():
    data_list = torch.load(PT_FILE, weights_only=False)

    print("Total graphs:", len(data_list))
    print("=" * 60)

    for i, data in enumerate(data_list[:5]):
        print(f"Graph #{i}")
        print(" rule_id:", data.rule_id)
        print(" y (label):", data.y.item())

        # ---- shapes + dtypes ----
        print(" x shape:", tuple(data.x.shape), "dtype:", data.x.dtype)
        print(" edge_index shape:", tuple(data.edge_index.shape), "dtype:", data.edge_index.dtype)
        print(" edge_attr shape:", tuple(data.edge_attr.shape), "dtype:", data.edge_attr.dtype)

        # ---- strict sanity checks (de Bruijn expected) ----
        assert tuple(data.x.shape) == (16, 4), "Expected x shape (16,4) for de Bruijn graph"
        assert tuple(data.edge_index.shape) == (2, 32), "Expected edge_index shape (2,32)"
        assert tuple(data.edge_attr.shape) == (32, 1), "Expected edge_attr shape (32,1)"
        assert data.edge_index.dtype == torch.long, "edge_index must be torch.long"
        assert data.y.dtype == torch.long, "y must be torch.long"
        assert data.x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), "x should be integer"
        assert data.edge_attr.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), "edge_attr should be integer"

        # ---- print first few nodes ----
        print(" x (first 5 nodes as bits):")
        for nid in range(5):
            print(f"  node {nid:2d} -> x={data.x[nid].tolist()} (id_bits={node_id_to_bits4(nid)})")

        # ---- print first 5 edges ----
        print(" first 5 edges:")
        for e in range(5):
            s = data.edge_index[0, e].item()
            t = data.edge_index[1, e].item()
            out_bit = int(data.edge_attr[e, 0].item())  # scalar 0/1

            print(f"  {s:2d}({node_id_to_bits4(s)}) -> {t:2d}({node_id_to_bits4(t)}), edge_attr(c')={out_bit}")

        print("-" * 60)

if __name__ == "__main__":
    main()