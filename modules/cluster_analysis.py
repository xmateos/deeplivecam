import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


#def find_cluster_centroids(embeddings, max_k=10) -> Any:
#    embeddings = normalize(np.array(embeddings))
#    inertia = []
#    cluster_centroids = []
#    K = range(1, max_k+1)
#
#    for k in K:
#        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
#        kmeans.fit(embeddings)
#        inertia.append(kmeans.inertia_)
#        cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})
#
#    diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
#    best_k_index = diffs.index(max(diffs)) + 1
#    optimal_centroids = cluster_centroids[best_k_index]['centroids']
#
#    return optimal_centroids


def find_cluster_centroids(embeddings, max_k=10, elbow_tolerance=0.05):
    embeddings = normalize(np.array(embeddings))
    inertia = []
    cluster_centroids = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
        cluster_centroids.append(kmeans.cluster_centers_)

    # Encuentra el primer "codo" donde la mejora relativa cae por debajo del umbral
    for i in range(1, len(inertia)):
        improvement = (inertia[i - 1] - inertia[i]) / inertia[i - 1]
        if improvement < elbow_tolerance:
            print(f"🧠 Seleccionado k={i} con mejora {improvement:.3f}")
            return cluster_centroids[i - 1]

    # Si no encontró "codo", usa el máximo
    print(f"🧠 No se detectó codo claro, usando k={max_k}")
    return cluster_centroids[-1]


#def find_closest_centroid(centroids: list, normed_face_embedding) -> list:
#    try:
#        centroids = np.array(centroids)
#        normed_face_embedding = np.array(normed_face_embedding)
#        similarities = np.dot(centroids, normed_face_embedding)
#        closest_centroid_index = np.argmax(similarities)
#
#        return closest_centroid_index, centroids[closest_centroid_index]
#    except ValueError:
#        return None

def find_closest_centroid(centroids, normed_face_embedding, threshold=0.35):
    """
    Devuelve el índice del centroide más cercano solo si la similitud coseno supera el umbral.
    """
    centroids = np.array(centroids)
    normed_face_embedding = np.array(normed_face_embedding)

    similarities = cosine_similarity([normed_face_embedding], centroids)[0]
    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]

    if best_score < threshold:
        return None, None  # No asignar si es demasiado diferente

    return best_index, best_score