import torch

data_list = torch.load("pyg_radius2_symbol01_dedup.pt", weights_only=False)

print("Total graphs:", len(data_list))
print("=" * 50)

for i, data in enumerate(data_list[:5]):
    print(f"Graph #{i}")
    print(" rule_id   :", data.rule_id)
    print(" y (label) :", data.y.item())
    print(" x shape   :", data.x.shape)
    print(" edge_index shape :", data.edge_index.shape)
    print(" edge_attr shape  :", data.edge_attr.shape)

    print(" x =")
    print(data.x)

    print(" first 5 edges (src -> dst):")
    for e in range(5):
        s = data.edge_index[0, e].item()
        t = data.edge_index[1, e].item()
        attr = data.edge_attr[e].tolist()
        print(f"   {s} -> {t}, edge_attr={attr}")

    print("-" * 50)