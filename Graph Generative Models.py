# First, I'm importing all the necessary libraries and modules. PyTorch is the backbone for our model's operations, 
# including tensor operations and neural network functions. 
# PyTorch Geometric (PyG) is used for graph neural network layers, specifically GCNConv for graph convolution and VGAE for variational graph autoencoders. 
# NetworkX and matplotlib are for visualizing the graphs.
    
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE as PyG_VGAE
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import networkx as nx

# Here, I define an Encoder class, a custom torch.nn.Module, for our variational graph autoencoder. 
# It takes input and output channel sizes, sets up graph convolutional layers for the base, mu, and logstd operations.
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.base_conv = GCNConv(in_channels, out_channels)  # Base convolution layer
        self.mu_conv = GCNConv(out_channels, out_channels)  # Mean convolution layer
        self.logstd_conv = GCNConv(out_channels, out_channels)  # Log standard deviation layer

    def forward(self, x, edge_index):
        x = F.relu(self.base_conv(x, edge_index))  # Apply ReLU activation after base conv
        return self.mu_conv(x, edge_index), self.logstd_conv(x, edge_index)  # Return the mean and logstd layers' outputs

# VGAEModel extends PyG's VGAE class, initializing with our custom Encoder. It's ready to be used for graph autoencoding tasks.
class VGAEModel(PyG_VGAE):
    def __init__(self, in_channels, out_channels):
        super(VGAEModel, self).__init__(encoder=Encoder(in_channels, out_channels), decoder=None)

# GraphDataset is a simple class to hold our graph data, including features (x), edge connections (edge_index), labels (y), and a training mask.
class GraphDataset:
    def __init__(self, x, edge_index, y, train_mask):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_mask = train_mask

# train_vgae is a function for training our variational graph autoencoder model. 
        # It takes the model, data, and optimizer as inputs, performs a forward pass, calculates the loss, and updates the model's weights.
        
def train_vgae(model, data, optimizer):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients
    z = model.encode(data.x, data.edge_index)  # Encode graph data to latent space
    loss = model.recon_loss(z, data.edge_index) + (1 / data.x.size(0)) * model.kl_loss()  # Calculate loss
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights
    return loss.item()

# visualize_graph is a utility function for visualizing graphs using NetworkX. It takes an edge_index tensor and an optional title.
def visualize_graph(edge_index, title="Graph"):
    G = nx.Graph()  # Create a new graph
    edge_list = edge_index.t().tolist()  # Convert edge_index to a list of edges
    G.add_edges_from(edge_list)  # Add edges to the graph
    nx.draw(G, with_labels=True, node_color='skyblue')  # Draw the graph with node labels and a specified color
    plt.title(title)  # Set the plot title
    plt.show()  # Display the plot

# The main function orchestrates the entire process: preparing the data, creating the model, training, and visualization.
def main():
    x = torch.eye(4)  # Node features: identity matrix for simplicity
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long)  # Edge connections
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)  # Node labels
    train_mask = torch.tensor([True, True, True, True], dtype=torch.bool)  # Training mask
    data = GraphDataset(x=x, edge_index=edge_index, y=y, train_mask=train_mask)  # Prepare data

    in_channels = x.size(1)  # Input channel size
    out_channels = 2  # Output channel size for the latent space
    vgae_model = VGAEModel(in_channels, out_channels)  # Create VGAE model
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.01)  # Optimizer

    for epoch in range(200):  # Training loop
        loss = train_vgae(vgae_model, data, optimizer)  # Train model
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")  # Print loss every 10 epochs

    visualize_graph(data.edge_index, "Original Graph")  # Visualize the original graph

    vgae_model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        z = vgae_model.encode(data.x, data.edge_index)  # Encode to latent space
        adj_matrix = vgae_model.decoder.forward_all(z)  # Decode latent space to adjacency matrix
        adj_matrix = (adj_matrix > 0.5).float()  # Threshold to get binary adjacency matrix
        new_edge_index, _ = dense_to_sparse(adj_matrix)  # Convert dense matrix to sparse edge_index format

    visualize_graph(new_edge_index, "Generated Graph")  # Visualize the generated graph

if __name__ == "__main__":
    main()  # Run the main function if this script is executed