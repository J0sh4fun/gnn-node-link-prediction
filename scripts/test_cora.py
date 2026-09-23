import torch
from torch_geometric.datasets import Planetoid


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Load Cora
dataset = Planetoid(
    root="data/Cora",
    name="Cora"
)

data = dataset[0]

print("\n=== Cora ===")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of node features:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)

print("\n=== Shapes ===")
print("x:", data.x.shape)
print("edge_index:", data.edge_index.shape)
print("y:", data.y.shape)

print("\n=== Masks ===")
print("train nodes:", data.train_mask.sum().item())
print("val nodes:", data.val_mask.sum().item())
print("test nodes:", data.test_mask.sum().item())

# Move graph to GPU
data = data.to(device)

print("\n=== GPU check ===")
print("x device:", data.x.device)
print("edge_index device:", data.edge_index.device)