# src/train.py
import torch
from torch_geometric.loader import DataLoader
from model import GNN
from graph_generation import build_graph

def train_model(train_data, val_data, epochs=50, lr=0.001):
    """Train the GNN model."""
    model = GNN(input_dim=train_data.num_node_features, hidden_dim=128, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out.squeeze(), train_data.y.float())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = criterion(val_out.squeeze(), val_data.y.float())
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

    return model
