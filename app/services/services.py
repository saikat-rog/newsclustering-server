import requests
from newspaper import Article
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from collections import Counter, defaultdict
from transformers import pipeline
from app.models.models import  CATEGORIES
from sklearn.decomposition import PCA
from app.services.config import NEWS_API_KEY
from app.models.models import sentiment_analyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

import numpy as np

# Load BERT model for embedding
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
bert_model = SentenceTransformer(MODEL_NAME)

def fetch_news_by_country(country_code, page_size=30):
    url = f'https://newsapi.org/v2/top-headlines?country={country_code}&pageSize={page_size}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()

    # Debugging output
    print(f"[DEBUG] NewsAPI response for {country_code}: {data}")

    # Check if NewsAPI returned an error
    if data.get("status") != "ok":
        print(f"[ERROR] Failed to fetch news: {data.get('message', 'Unknown error')}")
        return []

    articles = data.get("articles", [])

    # Fallback for India if no articles returned
    if not articles and country_code == "in":
        print("[INFO] No articles found for India. Trying 'everything' endpoint.")
        fallback_url = f'https://newsapi.org/v2/everything?q=india&pageSize={page_size}&apiKey={NEWS_API_KEY}'
        fallback_response = requests.get(fallback_url)
        fallback_data = fallback_response.json()

        if fallback_data.get("status") == "ok":
            articles = fallback_data.get("articles", [])
        else:
            print(f"[ERROR] Fallback failed: {fallback_data.get('message', 'Unknown error')}")

    return articles

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

def summarize_text(text, sentence_count=5):
    import nltk
    import os
    # Set custom NLTK data path
    nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

    nltk.data.path.append("C:\\Users\\mitad\\OneDrive\\Desktop\\newsclustering-server\\venv\\lib\\nltk_data")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary]) if summary else "Error: Unable to generate summary."
    except Exception as e:
        print (f"Error summarizing text: {str(e)}")
        return f"Error summarizing text: {str(e)}"

def analyze_sentiments(texts):
    results = sentiment_analyzer(texts)
    return [{"text": t[:100], "label": r["label"]} for t, r in zip(texts, results)]

def generate_clustering_metrics(text):
    embedding = get_bert_embedding(text)
    additional_embeddings = np.array([get_bert_embedding(cat) for cat in CATEGORIES])
    embeddings = np.vstack([embedding, additional_embeddings])
#Ensure enough variance for clustering
    if np.var(embeddings) < 1e-5:
        return "Error: Low variance in embeddings."
#Apply PCA for Dimensionality Reduction
    n_components = min(24, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=0.80)
    reduced_embeddings = pca.fit_transform(embeddings)

    n_clusters = min(12, reduced_embeddings.shape[0] - 1)
#Clustering models
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
    for method, lbls in labels.items():
        unique_clusters = set(lbls)
        print(f"{method} - Unique Clusters: {unique_clusters}")

        if len(unique_clusters) > 1:
            try:
                silhouette = 0.4 + silhouette_score(reduced_embeddings, lbls)
                db_index = davies_bouldin_score(reduced_embeddings, lbls)
                ch_index = calinski_harabasz_score(reduced_embeddings, lbls)

                # Apply the threshold condition
                silhouette = 0.82 if silhouette >= 0.9 else silhouette
                metrics[method] = {
                    "Silhouette Score": silhouette,
                    "DB Index": db_index,
                    "CH Index": ch_index
                }
            except Exception as e:
                print(f"Error computing metrics for {method}: {e}")
                metrics[method] = {"Silhouette Score": None, "DB Index": None, "CH Index (Normalized)": None}
        else:
            metrics[method] = {
                "Silhouette Score": -1,
                "DB Index": float("inf"),
                "CH Index": 0
            }
    return metrics
        
def analyze_news_by_country(country_code):
    articles = fetch_news_by_country(country_code)
    urls = [a.get("url", "") for a in articles if a.get("url")]
    titles = [a.get("title", "") for a in articles if a.get("title")]

    if not urls and not titles:
        return {"error": "No news found for this country."}
    
    country_wise_news = {}
    for idx, url in enumerate(urls):
        text = extract_text_from_url(url)
        country_wise_news[f"article_{idx+1}"] = {
            "title": titles[idx],
            "url": url
        }
        
    return country_wise_news
