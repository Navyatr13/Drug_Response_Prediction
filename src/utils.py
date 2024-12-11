# src/utils.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(embeddings, labels):
    """Visualize embeddings using PCA."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.title("Graph Embeddings Visualization")
    plt.show()
