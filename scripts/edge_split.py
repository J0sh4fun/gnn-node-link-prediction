import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling


SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.05


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. Load Cora
# ============================================================

dataset = Planetoid(
    root="./data",
    name="Cora",
)

data = dataset[0]

print("\n=== Original graph ===")
print("Nodes:", data.num_nodes)
print("Edges:", data.edge_index.size(1))


# ============================================================
# 3. Get unique undirected positive edges
# ============================================================

edge_index = data.edge_index.to(device)

src = edge_index[0]
dst = edge_index[1]

mask = src < dst

positive_edges = edge_index[:, mask]

num_positive = positive_edges.size(1)

print("\n=== Positive edges ===")
print("Total:", num_positive)


# ============================================================
# 4. Shuffle positive edges
# ============================================================

generator = torch.Generator(device=device)
generator.manual_seed(SEED)

perm = torch.randperm(
    num_positive,
    generator=generator,
    device=device,
)

positive_edges = positive_edges[:, perm]


# ============================================================
# 5. Split positive edges
# ============================================================

num_train = int(num_positive * TRAIN_RATIO)
num_val = int(num_positive * VAL_RATIO)

train_pos = positive_edges[:, :num_train]

val_pos = positive_edges[
    :, num_train:num_train + num_val
]

test_pos = positive_edges[
    :, num_train + num_val:
]


print("\n=== Positive split ===")
print("Train:", train_pos.size(1))
print("Validation:", val_pos.size(1))
print("Test:", test_pos.size(1))


# ============================================================
# 6. Generate ONE global negative pool
# ============================================================
#
# Generate enough negatives for ALL three splits at once.
#
# IMPORTANT:
# We use the ORIGINAL graph here.
#
# This prevents a real edge that happens to be held out
# for validation/test from being selected as negative.
# ============================================================

total_negative = (
    train_pos.size(1)
    + val_pos.size(1)
    + test_pos.size(1)
)

negative_edges = negative_sampling(
    edge_index=edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=total_negative,
    force_undirected=True,
)

print("\n=== Global negative pool ===")
print("Total:", negative_edges.size(1))


# ============================================================
# 7. Split negative pool
# ============================================================

train_neg = negative_edges[:, :train_pos.size(1)]

val_start = train_pos.size(1)
val_end = val_start + val_pos.size(1)

val_neg = negative_edges[:, val_start:val_end]

test_neg = negative_edges[:, val_end:]


print("\n=== Negative split ===")
print("Train:", train_neg.size(1))
print("Validation:", val_neg.size(1))
print("Test:", test_neg.size(1))


# ============================================================
# 8. Create edge labels
# ============================================================

train_edge_label_index = torch.cat(
    [train_pos, train_neg],
    dim=1,
)

train_labels = torch.cat(
    [
        torch.ones(train_pos.size(1), device=device),
        torch.zeros(train_neg.size(1), device=device),
    ]
)


val_edge_label_index = torch.cat(
    [val_pos, val_neg],
    dim=1,
)

val_labels = torch.cat(
    [
        torch.ones(val_pos.size(1), device=device),
        torch.zeros(val_neg.size(1), device=device),
    ]
)


test_edge_label_index = torch.cat(
    [test_pos, test_neg],
    dim=1,
)

test_labels = torch.cat(
    [
        torch.ones(test_pos.size(1), device=device),
        torch.zeros(test_neg.size(1), device=device),
    ]
)


# ============================================================
# 9. Convert edges to sets for validation
# ============================================================

def edge_set(edge_index):
    return {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.t().tolist()
    }


original_set = edge_set(positive_edges)

train_pos_set = edge_set(train_pos)
val_pos_set = edge_set(val_pos)
test_pos_set = edge_set(test_pos)

train_neg_set = edge_set(train_neg)
val_neg_set = edge_set(val_neg)
test_neg_set = edge_set(test_neg)


# ============================================================
# 10. Check positive split overlap
# ============================================================

print("\n=== Positive overlap check ===")

print(
    "Train ∩ Val:",
    len(train_pos_set & val_pos_set),
)

print(
    "Train ∩ Test:",
    len(train_pos_set & test_pos_set),
)

print(
    "Val ∩ Test:",
    len(val_pos_set & test_pos_set),
)


# ============================================================
# 11. Check negative edges are NOT real edges
# ============================================================

print("\n=== Negative vs original graph ===")

print(
    "Train negative ∩ Original:",
    len(train_neg_set & original_set),
)

print(
    "Val negative ∩ Original:",
    len(val_neg_set & original_set),
)

print(
    "Test negative ∩ Original:",
    len(test_neg_set & original_set),
)


# ============================================================
# 12. Check negative split overlap
# ============================================================

print("\n=== Negative overlap check ===")

print(
    "Train ∩ Val:",
    len(train_neg_set & val_neg_set),
)

print(
    "Train ∩ Test:",
    len(train_neg_set & test_neg_set),
)

print(
    "Val ∩ Test:",
    len(val_neg_set & test_neg_set),
)


# ============================================================
# 13. Final shapes
# ============================================================

print("\n=== Final data ===")

print("Train:")
print("  edge_label_index:", train_edge_label_index.shape)
print("  labels:", train_labels.shape)

print("\nValidation:")
print("  edge_label_index:", val_edge_label_index.shape)
print("  labels:", val_labels.shape)

print("\nTest:")
print("  edge_label_index:", test_edge_label_index.shape)
print("  labels:", test_labels.shape)