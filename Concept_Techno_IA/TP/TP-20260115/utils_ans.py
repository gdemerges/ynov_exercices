import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

def setup_environment():
    """Configure l'environnement numpy et matplotlib."""
    np.set_printoptions(threshold=10000, suppress=True)
    import warnings
    warnings.filterwarnings('ignore')


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_data(filepath, sep=';'):
    data = pd.read_csv(filepath, sep=sep)
    print(f"Dimensions : {data.shape}")
    print(f"Colonnes : {list(data.columns)}")
    print("\nAperçu :")
    print(data.head())
    return data


def prepare_data(data, feature_cols, label_col=0):
    if isinstance(feature_cols, list):
        X = data[feature_cols].values
        feature_names = feature_cols
    else:
        X = data.iloc[:, feature_cols].values
        feature_names = list(data.columns[feature_cols])

    if isinstance(label_col, int):
        labels = data.iloc[:, label_col].values
    else:
        labels = data[label_col].values

    print(f"Matrice X : {X.shape}")
    print(f"Labels : {labels.shape}")
    print(f"Features : {feature_names}")

    return X, labels, feature_names


# =============================================================================
# ACP - ANALYSE EN COMPOSANTES PRINCIPALES
# =============================================================================

def perform_pca(X, n_components=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_scaled, X_pca, pca, scaler


def print_variance_explained(pca, n_display=10):
    print("Variance expliquée par chaque composante :")
    n = min(n_display, len(pca.explained_variance_ratio_))
    for i, var in enumerate(pca.explained_variance_ratio_[:n]):
        print(f"  PC{i+1}: {var*100:.2f}%")

    print("\nVariance cumulée :")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, var in enumerate(cumulative[:n]):
        print(f"  PC1-PC{i+1}: {var*100:.2f}%")

    return cumulative


def plot_variance(pca, figsize=(12, 5)):
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_components = len(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Variance par composante
    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_ * 100, color='steelblue')
    axes[0].set_xlabel('Composante principale')
    axes[0].set_ylabel('Variance expliquée (%)')
    axes[0].set_title('Variance par composante')
    if n_components <= 12:
        axes[0].set_xticks(range(1, n_components + 1))

    # Variance cumulée
    axes[1].plot(range(1, n_components + 1), cumulative * 100, 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=80, color='r', linestyle='--', label='80% variance')
    axes[1].axhline(y=95, color='g', linestyle='--', label='95% variance')
    axes[1].set_xlabel('Nombre de composantes')
    axes[1].set_ylabel('Variance cumulée (%)')
    axes[1].set_title('Variance cumulée')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return cumulative


def get_loadings(pca, feature_names, n_components=3):
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    n = min(n_components, loadings.shape[1])
    cols = [f'PC{i+1}' for i in range(n)]
    loadings_df = pd.DataFrame(loadings[:, :n], columns=cols, index=feature_names)

    print("Corrélations des variables avec les axes principaux :")
    print(loadings_df.round(3))

    return loadings_df


def plot_correlation_circle(pca, feature_names, figsize=(8, 8)):
    plt.figure(figsize=figsize)

    # Cercle unité
    circle = plt.Circle((0, 0), 1, fill=False, color='blue', linewidth=2)
    plt.gca().add_patch(circle)

    # Loadings normalisés
    loadings_norm = pca.components_[:2, :].T

    for i, var in enumerate(feature_names):
        plt.arrow(0, 0, loadings_norm[i, 0], loadings_norm[i, 1],
                  head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
        plt.text(loadings_norm[i, 0] * 1.15, loadings_norm[i, 1] * 1.15, var,
                 ha='center', va='center', fontsize=9)

    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Cercle des corrélations')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.show()


# =============================================================================
# CLUSTERING
# =============================================================================

def apply_kmeans(X, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clustering = kmeans.fit_predict(X)
    return clustering, kmeans


def apply_agglomerative(X, n_clusters=3, linkage='ward'):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clustering = agg.fit_predict(X)
    return clustering


def clustering_hybride(X, n_clusters=3, random_state=42):
    # Étape 1 : Classification hiérarchique Ward
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_init = agg.fit_predict(X)

    # Calcul des centroïdes initiaux
    initial_centers = np.array([X[labels_init == i].mean(axis=0) for i in range(n_clusters)])

    # Étape 2 : KMeans avec initialisation intelligente
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, random_state=random_state)
    final_labels = kmeans.fit_predict(X)

    return final_labels, kmeans.cluster_centers_


def plot_clustering(X_pca, clustering, labels, title, n_clusters=3,
                    figsize=(12, 8), annotate=True):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    plt.figure(figsize=figsize)

    for i in range(n_clusters):
        mask = clustering == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i % len(colors)], label=f'Cluster {i} (n={mask.sum()})',
                   s=80, alpha=0.7)

    if annotate:
        for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
            plt.annotate(str(label), xy=(x, y), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# ÉVALUATION
# =============================================================================

def compute_silhouette_scores(X, k_range=range(2, 10), random_state=42):
    scores = []
    print("Indices Silhouette pour KMeans :")
    print("=" * 40)

    for k in k_range:
        clustering = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        score = metrics.silhouette_score(X, clustering, metric='euclidean')
        scores.append(score)
        print(f"K={k}: Silhouette = {score:.4f}")

    best_k = list(k_range)[np.argmax(scores)]
    best_score = max(scores)
    print("=" * 40)
    print(f"\nMeilleure partition : K={best_k} (Silhouette = {best_score:.4f})")

    return scores, best_k


def plot_silhouette_scores(scores, k_range=range(2, 10), figsize=(10, 6)):
    k_list = list(k_range)
    best_k = k_list[np.argmax(scores)]
    best_score = max(scores)

    plt.figure(figsize=figsize)
    plt.plot(k_list, scores, 'bo-', linewidth=2, markersize=10)
    plt.xlabel('Nombre de clusters (K)', fontsize=12)
    plt.ylabel('Indice Silhouette', fontsize=12)
    plt.title('Indice Silhouette en fonction du nombre de clusters', fontsize=14)
    plt.xticks(k_list)
    plt.grid(True, alpha=0.3)

    # Marquer le maximum
    plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Meilleur K={best_k}')
    plt.scatter([best_k], [best_score], color='red', s=200, zorder=5)
    plt.legend(fontsize=11)
    plt.show()


def compare_methods(X_scaled, n_clusters=3, random_state=42):
    results = {}

    # KMeans
    clustering_km, _ = apply_kmeans(X_scaled, n_clusters, random_state)
    score_km = metrics.silhouette_score(X_scaled, clustering_km, metric='euclidean')
    results['KMeans'] = (clustering_km, score_km)

    # Single
    clustering_single = apply_agglomerative(X_scaled, n_clusters, 'single')
    score_single = metrics.silhouette_score(X_scaled, clustering_single, metric='euclidean')
    results['Single'] = (clustering_single, score_single)

    # Ward
    clustering_ward = apply_agglomerative(X_scaled, n_clusters, 'ward')
    score_ward = metrics.silhouette_score(X_scaled, clustering_ward, metric='euclidean')
    results['Ward'] = (clustering_ward, score_ward)

    # Average
    clustering_avg = apply_agglomerative(X_scaled, n_clusters, 'average')
    score_avg = metrics.silhouette_score(X_scaled, clustering_avg, metric='euclidean')
    results['Average'] = (clustering_avg, score_avg)

    # Hybride
    clustering_hyb, _ = clustering_hybride(X_scaled, n_clusters, random_state)
    score_hyb = metrics.silhouette_score(X_scaled, clustering_hyb, metric='euclidean')
    results['Hybride'] = (clustering_hyb, score_hyb)

    # Affichage
    print(f"Comparaison des méthodes pour {n_clusters} clusters :")
    print("=" * 50)
    for method, (_, score) in results.items():
        print(f"{method:15} : Silhouette = {score:.4f}")
    print("=" * 50)

    best_method = max(results, key=lambda x: results[x][1])
    print(f"\nMeilleure méthode : {best_method} (Silhouette = {results[best_method][1]:.4f})")

    return results
