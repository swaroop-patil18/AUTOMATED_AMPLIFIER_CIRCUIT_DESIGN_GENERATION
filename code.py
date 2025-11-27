
circuit_vae.py



import math
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_NODES = 6           # maximum nodes in a graph
NODE_TYPES = ["R", "C", "L"]  # simple two-terminal devices for demo
NODE_TYPE_COUNT = len(NODE_TYPES)
LATENT_DIM = 16
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
EDGE_THRESHOLD = 0.5    # threshold to decide whether an edge exists after decoding

# ---------- Synthetic dataset ----------
def make_random_circuit_graph(num_nodes: int) -> nx.Graph:
    """Create a small random graph representing a circuit.
       Nodes carry a device type as attribute 'dtype' and a value (ohm/F/H) in 'val'."""
    G = nx.Graph()
    for n in range(num_nodes):
        dtype = random.choice(NODE_TYPES)
        # assign a random nominal value (for R: 10-1k, for C: pF-nF, for L: uH-mH)
        if dtype == "R":
            val = round(random.uniform(10, 1e3), 2)
        elif dtype == "C":
            val = round(random.uniform(1e-12, 1e-8), 12)
        else:
            val = round(random.uniform(1e-6, 1e-3), 9)
        G.add_node(n, dtype=dtype, val=val)
    # Connect nodes randomly, ensure connectivity
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < 0.3:
                G.add_edge(i, j)
    if not nx.is_connected(G):
        # make a spanning path if disconnected
        nodes = list(G.nodes)
        for i in range(len(nodes)-1):
            G.add_edge(nodes[i], nodes[i+1])
    return G

def graph_to_tensor(G: nx.Graph, max_nodes=MAX_NODES):
    """Convert graph to adjacency matrix (max_nodes x max_nodes) and node-type one-hot (max_nodes x type_count)."""
    n = G.number_of_nodes()
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    node_feat = np.zeros((max_nodes, NODE_TYPE_COUNT), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if G.has_edge(i, j):
                adj[i, j] = 1.0
    for i in range(n):
        dtype = G.nodes[i]["dtype"]
        idx = NODE_TYPES.index(dtype)
        node_feat[i, idx] = 1.0
    # Note: leftover nodes remain zeros (padding)
    return adj, node_feat, n

def create_dataset(n_graphs=2000):
    data = []
    for _ in range(n_graphs):
        n = random.randint(2, MAX_NODES)
        G = make_random_circuit_graph(n)
        adj, node_feat, n_nodes = graph_to_tensor(G)
        data.append((adj, node_feat, n_nodes))
    return data

# ---------- VAE model ----------
class GraphVAE(nn.Module):
    def __init__(self, max_nodes=MAX_NODES, node_types=NODE_TYPE_COUNT, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_types = node_types
        input_size = max_nodes*max_nodes + max_nodes*node_types

        # encoder
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        out = self.fc_out(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

# ---------- Utilities ----------
def pack_graph(adj_np, node_feat_np):
    """Flatten adjacency and node_feat into single vector."""
    return np.concatenate([adj_np.flatten(), node_feat_np.flatten()]).astype(np.float32)

def unpack_output(out_vec):
    """Split decoder output into adjacency logits and node feature logits."""
    total_adj = MAX_NODES*MAX_NODES
    adj_logits = out_vec[:, :total_adj]
    node_logits = out_vec[:, total_adj:]
    adj_logits = adj_logits.view(-1, MAX_NODES, MAX_NODES)
    node_logits = node_logits.view(-1, MAX_NODES, NODE_TYPE_COUNT)
    return adj_logits, node_logits

def loss_function(adj_logits, node_logits, adj_target, node_target, mu, logvar):
    # adjacency reconstruction: BCE (we treat edges as symmetric; ensure symmetric target)
    adj_loss = F.binary_cross_entropy_with_logits(adj_logits, adj_target, reduction='sum')
    # node type: BCE per one-hot (or could use cross entropy per node where node exists)
    node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (adj_loss + node_loss) / adj_target.size(0) + 1e-3 * kld, adj_loss, node_loss, kld

# ---------- Training ----------
def train(model, dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    data_vectors = [pack_graph(adj, feat) for adj, feat, _ in dataset]
    data_t = torch.tensor(np.stack(data_vectors, axis=0), dtype=torch.float32).to(DEVICE)
    n_samples = data_t.size(0)
    for epoch in range(1, epochs+1):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch = data_t[idx]
            optimizer.zero_grad()
            out, mu, logvar = model(batch)
            adj_logits, node_logits = unpack_output(out)
            # targets
            adj_target = batch[:, :MAX_NODES*MAX_NODES].view(-1, MAX_NODES, MAX_NODES)
            node_target = batch[:, MAX_NODES*MAX_NODES:].view(-1, MAX_NODES, NODE_TYPE_COUNT)
            loss, a_loss, n_loss, kld = loss_function(adj_logits, node_logits, adj_target, node_target, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.size(0)
        avg_loss = epoch_loss / n_samples
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} - Loss {avg_loss:.4f}")
    return model

# ---------- Decode latent vector to graph ----------
def decode_to_graph(adj_logits_np, node_logits_np, threshold=EDGE_THRESHOLD):
    """Take numpy arrays of logits (or probs) and convert to networkx Graph."""
    # If logits, convert via sigmoid and softmax
    adj_probs = 1/(1+np.exp(-adj_logits_np))
    node_probs = np.exp(node_logits_np) / np.sum(np.exp(node_logits_np), axis=-1, keepdims=True)
    # threshold adjacency and make symmetric
    adj_mat = (adj_probs > threshold).astype(int)
    # Symmetrize
    adj_mat = np.triu(adj_mat, 1)
    adj_mat = adj_mat + adj_mat.T
    G = nx.Graph()
    # determine used nodes (non-zero node type)
    for i in range(MAX_NODES):
        if node_probs[i].sum() < 1e-6:
            continue
        dtype_idx = int(np.argmax(node_probs[i]))
        dtype = NODE_TYPES[dtype_idx]
        # dummy value
        val = None
        G.add_node(i, dtype=dtype, val=val)
    # add edges
    nodes_present = list(G.nodes)
    for i in nodes_present:
        for j in nodes_present:
            if i < j and adj_mat[i, j] == 1:
                G.add_edge(i, j)
    # if G empty or single node, nothing to do
    if G.number_of_nodes() <= 1 and len(nodes_present) >= 2:
        # connect sequentially to form something
        for i in range(len(nodes_present)-1):
            G.add_edge(nodes_present[i], nodes_present[i+1])
    return G

# ---------- SPICE export ----------
def graph_to_spice_netlist(G: nx.Graph, filename="generated_circuit.spice"):
    """Simple 2-terminal device netlist:
       R1 n1 n2 1k
       etc.
       Adds V1 n0 0 DC 1.0 as a reference source for simulation.
    """
    lines = ["* Generated SPICE-like netlist"]
    lines.append("V1 0 n0 DC 1.0")  # a reference source connected to node 0
    dev_count = 1
    for (u, v, data) in G.edges(data=True):
        # choose device type from one of the connected nodes (simple heuristic)
        dtype_u = G.nodes[u].get("dtype", "R")
        dtype_v = G.nodes[v].get("dtype", "R")
        # pick majority or first
        dtype = dtype_u
        name = f"{dtype}{dev_count}"
        # choose a default value
        if dtype == "R":
            val = "1k"
        elif dtype == "C":
            val = "1n"
        elif dtype == "L":
            val = "1u"
        else:
            val = "1k"
        lines.append(f"{name} n{u} n{v} {val}")
        dev_count += 1
    lines.append(".end")
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    return filename

# ---------- Inference / Sampling ----------
def sample_from_model(model: GraphVAE, n_samples=5):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            out = model.decode(z)
            adj_logits, node_logits = unpack_output(out)
            adj_np = adj_logits.cpu().numpy().squeeze(0)
            node_np = node_logits.cpu().numpy().squeeze(0)
            G = decode_to_graph(adj_np, node_np)
            samples.append(G)
    return samples

# ---------- Demo ----------
def demo_run():
    print("Preparing dataset...")
    dataset = create_dataset(1500)
    print(f"Dataset size: {len(dataset)} graphs")
    model = GraphVAE().to(DEVICE)
    print("Training VAE (this is a small toy training)...")
    model = train(model, dataset, epochs=60, batch_size=128)
    print("Sampling graphs from latent space...")
    samples = sample_from_model(model, n_samples=6)
    for idx, G in enumerate(samples):
        print(f"\nSample #{idx+1}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        for n, d in G.nodes(data=True):
            print(f"  Node {n}: type={d.get('dtype')}")
        # export to spice
        fname = f"sample_{idx+1}.spice"
        graph_to_spice_netlist(G, filename=fname)
        print(f"  Exported SPICE netlist to {fname}")

if __name__ == "__main__":
    demo_run()
