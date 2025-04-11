import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

def load_training_data():
    # Replace with your actual feature extraction process (e.g., BERT embeddings)
    # For demo, generate fake data
    X = np.load('data/bert_embeddings.npy') # Simulated 200 news items with 50 features
    return StandardScaler().fit_transform(X)

def evaluate_model(X, labels):
    return {
        "Silhouette Score": silhouette_score(X, labels),
        "DB Index": davies_bouldin_score(X, labels),
        "CH Index": calinski_harabasz_score(X, labels),
    }

def retrain_kmeans(X):
    print("🔁 Retraining KMeans...")
    model = KMeans(n_clusters=24, n_init=20, random_state=42)
    labels = model.fit_predict(X)
    joblib.dump(model, "models/kmeans_model.pkl")
    return evaluate_model(X, labels)

def retrain_agnes(X):
    print("🔁 Retraining AGNES...")
    Z = linkage(X, method='ward')
    labels = fcluster(Z, 24, criterion='maxclust')
    return evaluate_model(X, labels)

def retrain_spectral(X):
    print("🔁 Retraining Spectral Clustering...")
    model = SpectralClustering(n_clusters=24, affinity='nearest_neighbors', random_state=42)
    labels = model.fit_predict(X)
    joblib.dump(model, "models/spectral_model.pkl")
    return evaluate_model(X, labels)

def retrain_gmm(X):
    print("🔁 Retraining GMM...")
    model = GaussianMixture(n_components=24, random_state=42)
    labels = model.fit_predict(X)
    joblib.dump(model, "models/gmm_model.pkl")
    return evaluate_model(X, labels)

def process_feedback():
    print("📥 Reading feedback...")
    # Read raw CSV
    df = pd.read_csv('feedback_training.csv')
    # Convert to proper booleans
    df['all_algorithms'] = df['all_algorithms'].astype(str).str.strip().str.lower().map({'true': True, 'false': False})


    # Optional: Save it back
    df.to_csv('feedback_training.csv', index=False)

    X = load_training_data()

    preferred_counts = df['preferred_algorithm'].value_counts()
    disliked_all = df[df['all_algorithms'] == True]

    results = {}

    if not preferred_counts.empty:
        top_algo = preferred_counts.idxmax()
        print(f"✨ Most liked algorithm: {top_algo}")

        if top_algo == 'KMeans':
            results['KMeans'] = retrain_kmeans(X)
        elif top_algo == 'AGNES':
            results['AGNES'] = retrain_agnes(X)
        elif top_algo == 'Spectral':
            results['Spectral'] = retrain_spectral(X)
        elif top_algo == 'GMM':
            results['GMM'] = retrain_gmm(X)

    # If no preferred algorithm given OR user disliked all
        if preferred_counts.empty or not disliked_all.empty:
            print("👥 No preference or negative feedback — retraining all algorithms.")
            results['KMeans'] = retrain_kmeans(X)
            results['AGNES'] = retrain_agnes(X)
            results['Spectral'] = retrain_spectral(X)
            results['GMM'] = retrain_gmm(X)
            print("\n📊 Updated Performance Scores:")
    for algo, scores in results.items():
        print(f"\n🔸 {algo}")
        for metric, score in scores.items():
            print(f"   {metric}: {score:.4f}")

    return results
