import torch, random

def count_by_class(labels, C=8):
    cnt = [0]*C
    for k in labels:
        cnt[int(k)] += 1
    return cnt

pt = "output/pyg_radius2_debruijn_dedup.pt"
graphs = torch.load(pt, map_location="cpu", weights_only=False)

labels = [int(g.y) if not hasattr(g.y, "item") else int(g.y.item()) for g in graphs]
print("ALL:", count_by_class(labels, 8))

# random split demo
idx = list(range(len(labels)))
random.seed(42)
random.shuffle(idx)
n = len(idx)
n_test = int(0.2*n)
n_val  = int(0.1*n)

test  = idx[:n_test]
val   = idx[n_test:n_test+n_val]
train = idx[n_test+n_val:]

print("TRAIN:", count_by_class([labels[i] for i in train], 8))
print("VAL  :", count_by_class([labels[i] for i in val], 8))
print("TEST :", count_by_class([labels[i] for i in test], 8))