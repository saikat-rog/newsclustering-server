import requests
from newspaper import Article
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from collections import Counter, defaultdict
from transformers import pipeline
from app.models.models import get_bert_embedding, CATEGORIES
from sklearn.decomposition import PCA
from app.services.config import NEWS_API_KEY
import numpy as np

# Load BERT model for embedding
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
bert_model = SentenceTransformer(MODEL_NAME)
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

def fetch_news_by_country(country_code, page_size=30):
    url = f'https://newsapi.org/v2/top-headlines?country={country_code}&pageSize={page_size}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    return response.json().get("articles", [])



def get_bert_embedding(text):
    return bert_model.encode(text)

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text if article.text else "Error: No text extracted."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def safe_summarize(text):
    word_count = len(text.split())
    max_len = max(20, int(word_count * 0.4))
    min_len = max(10, int(word_count * 0.2))
    try:
        return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
    except:
        return text[:150]

def summarize_text(text):
    try:
        summary = safe_summarize(text)
        return summary[0]['summary_text']
    except:
        return text[:150]  # fallback

def perform_all_clusterings(texts, n_clusters=4):
    embeddings = [get_bert_embedding(t) for t in texts]

    # Clustering
    kmeans_labels = KMeans(n_clusters=n_clusters).fit_predict(embeddings)
    gmm_labels = GaussianMixture(n_components=n_clusters).fit_predict(embeddings)
    agnes_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    spectral_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit_predict(embeddings)

    # Organize cluster text
    cluster_results = {}
    for name, labels in zip(["kmeans", "gmm", "agnes", "spectral"], [kmeans_labels, gmm_labels, agnes_labels, spectral_labels]):
        clusters = [
            {
                "text": t[:100],
                "cluster": int(label),
                "summary": summarize_text(t),
                "category": f"Cluster {label}"
            } for t, label in zip(texts, labels)
        ]
        cluster_results[name] = clusters

    return cluster_results

def analyze_sentiments(texts):
    results = sentiment_analyzer(texts)
    return [{"text": t[:100], "label": r["label"]} for t, r in zip(texts, results)]

def generate_clustering_metrics(text):
    embedding = get_bert_embedding(text)
    additional_embeddings = np.array([get_bert_embedding(cat) for cat in CATEGORIES])
    embeddings = np.vstack([embedding, additional_embeddings])

    if np.var(embeddings) < 1e-5:
        return "Error: Low variance in embeddings."

    n_components = min(15, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    n_clusters = min(7, reduced_embeddings.shape[0] - 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(reduced_embeddings)
    agnes = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(reduced_embeddings)

    try:
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels="discretize").fit(reduced_embeddings)
        spectral_labels = spectral.labels_
    except Exception:
        spectral_labels = np.zeros(reduced_embeddings.shape[0])

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42).fit(reduced_embeddings)

    labels = {
        "K-Means": kmeans.labels_,
        "AGNES": agnes.labels_,
        "Spectral Clustering": spectral_labels,
        "GMM Clustering": gmm.predict(reduced_embeddings)
    }

    metrics = {}
    for name, labels in labels.items():
        if len(set(labels)) > 1:  # Need at least 2 clusters to compute metrics
            metrics[name] = {
                "Silhouette Score": silhouette_score(reduced_embeddings, labels),
                "CH Index": calinski_harabasz_score(reduced_embeddings, labels),
                "DB Index": davies_bouldin_score(reduced_embeddings, labels)
            }
        else:
            metrics[name] = "Insufficient clusters for metric calculation."
            
    return metrics
    
def generate_trend_data(clusterings, sentiments):
    trend_data = {}
    for algo_name, clusters in clusterings.items():
        cluster_counts = Counter([item["cluster"] for item in clusters])
        trend_data[f"{algo_name}_distribution"] = dict(cluster_counts)

    sentiment_counts = Counter([item["label"] for item in sentiments])
    trend_data["sentiment_distribution"] = dict(sentiment_counts)

    return trend_data

def analyze_news_by_country(country_code):
    articles = fetch_news_by_country(country_code)
    contents = [a.get("content", "") for a in articles if a.get("content")]
    print(contents)

    if not contents:
        return {"error": "No news content found for this country."}

    clusterings = perform_all_clusterings(contents)
    sentiments = analyze_sentiments(contents)
    trends = generate_trend_data(clusterings, sentiments)

    return {
        "clusters": clusterings,
        "sentiments": sentiments,
        "trends": trends
    }
